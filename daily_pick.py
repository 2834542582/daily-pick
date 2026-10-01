#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每日选股推送 - 主线增强版 v6.3
===========================================================
v6.3 相对 v6.2 的改动：
  【数据源双备份】
  1) K线和资金流改为「东财优先 → 新浪回退」双源架构
     - 海外 GitHub Actions 环境自动走新浪（东财对数据中心 IP 有地理拦截）
     - 国内环境优先走东财（更准，含资金流细分字段）
  2) 探针逻辑重写：探针成功即继续，失败才降级
     - 不再"探针失败1次就熔断"，兼容"东财挂但新浪通"的场景

  【保留 v6.2 特性】
  - 评分归一化（修 135 vs 100 并列 bug）
  - 市场温度择时（涨停+梯队+炸板率）
  - 新浪行业源板块名修复
  - 弱势主线标签（🌡️）
  - hard_filter 单位修复（元 vs 亿）
  - 非交易日日期标签修复
  - 全部熔断/降级/超时保护

===========================================================
数据口径速查：
-----------------------------------------------------------
【涨停池 stock_zt_pool_em / 东财】
  连板数含当日（首板=1）；换手率为当日（%）；
  流通/总市值单位【元】；封板资金单位【元】；
  首次封板时间为 HHMMSS 六位数字；炸板次数=0 表示未开板。

【资金流 双源】
  东财 stock_individual_fund_flow：主力净流入 = 超大单+大单（元）
  新浪 MoneyFlow.ssl_qsfx_zjlrqs：字段 netamount = 主力净流入（元）
  本脚本用「近5日累计 / 近10日累计」，单位【元】

【K线 双源】
  东财 stock_zh_a_hist：前复权（adjust="qfq"）
  新浪 CN_MarketData.getKLineData：不复权（scale=240 日线）
  短期（60日）内不复权与复权差异极小，均线判断可接受

【板块主线】
  push2     : f3=当日涨幅(%)，f62=当日主力净流入(元)
  新浪行业  : 涨跌幅(%) + 成交额(万元)，无资金流
  同花顺/东财 akshare : 涨跌幅 + 主力净流入
  兜底      : 仅涨停家数 + 封板资金

【板块标签】
  🔥强势主线 : 强度分≥60 且（无资金或资金>0）
  📈升温中   : 40≤强度分<60
  🌡️弱势主线 : 10≤强度分<40
  📉走弱     : 强度分<10 且 资金<0
  ⚠️脉冲     : 涨幅>3 且 资金<0

【市场温度】
  涨停家数分（满分 40）+ 连板梯队分（满分 30）+ 炸板率分（满分 30）
  ≥75 热     : 满仓
  55~75 温   : 半仓
  35~55 冷   : 轻仓
  <35 冰冻   : 建议空仓

【评分权重】
  正常模式最高 135，降级模式最高 110，最终归一化到 100 分制；
  阈值：正常 75 分，降级 55 分。
===========================================================
"""
# ==================== 全局 requests 超时补丁 ====================
import requests

_ORIG_SESSION_REQUEST = requests.Session.request
_TIMEOUT = (3, 8)


def _patched_session_request(self, method, url, **kwargs):
    if kwargs.get("timeout") is None:
        kwargs["timeout"] = _TIMEOUT
    return _ORIG_SESSION_REQUEST(self, method, url, **kwargs)


requests.Session.request = _patched_session_request

_ORIG_API_REQUEST = requests.api.request


def _patched_api_request(method, url, **kwargs):
    if kwargs.get("timeout") is None:
        kwargs["timeout"] = _TIMEOUT
    return _ORIG_API_REQUEST(method, url, **kwargs)


requests.api.request = _patched_api_request
requests.get = lambda url, **kw: _patched_api_request("get", url, **kw)
requests.post = lambda url, **kw: _patched_api_request("post", url, **kw)
# =================================================================

import argparse
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime, timedelta
from typing import List

import pandas as pd
import numpy as np

# ==================== 配置 ====================
DEFAULT_TOP_N = 15
RETRY_COUNT = 2
RETRY_DELAY = 2
FUND_FAIL_THRESHOLD = 2
KL_INE_FAIL_THRESHOLD = 3
DEFAULT_MIN_SCORE = 75
DEGRADED_MIN_SCORE = 55
PROBE_N = 1

SCORE_TIME_BUDGET = 60
SLOW_CALL_THRESHOLD = 3.0
SLOW_CALL_MAX = 5

FALLBACK_STRONG_CNT = 3
FALLBACK_STRONG_CNT_LOW = 2
FALLBACK_TOTAL_ZT_LOW = 30
FALLBACK_WEAK_SEAL = 2e8

WEAK_STRENGTH_LOW = 10
WEAK_STRENGTH_HIGH = 40

TEMP_HOT = 75
TEMP_WARM = 55
TEMP_COLD = 35

HARD_FILTERS = {
    "st": True,
    "max_boards": 3,
    "max_turnover": 28.0,
    "one_word": True,
    "min_market_cap": 15.0,
    "late_afternoon": True,
    "seal_ratio_one_word": 0.08,
}

WEIGHTS_NORMAL = {
    "board_count": 5, "sector_main": 20, "fund_flow_5d": 15, "fund_flow_10d": 10,
    "morning_lobby": 10, "no_open": 10, "ma_bull": 15, "return_shot": 15,
    "market_cap_bonus": 5, "turnover_good": 5, "seal_amount": 5, "limit_gene": 10,
}
WEIGHTS_DEGRADED = {
    "board_count": 5, "sector_main": 30, "fund_flow_5d": 0, "fund_flow_10d": 0,
    "morning_lobby": 20, "no_open": 15, "ma_bull": 0, "return_shot": 0,
    "market_cap_bonus": 10, "turnover_good": 10, "seal_amount": 10, "limit_gene": 0,
}

# ==================== 全局熔断状态 ====================
_fund_circuit_broken = False    # 东财资金流熔断（仍可尝试新浪）
_fund_fail_count = 0
_kline_circuit_broken = False   # 东财K线熔断（仍可尝试新浪）
_kline_fail_count = 0
_fund_fast_fail = False         # 两个源都失败，完全熔断
_kline_fast_fail = False
_slow_fund_calls = 0
_slow_kline_calls = 0

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
       "AppleWebKit/537.36 (KHTML, like Gecko) "
       "Chrome/120.0.0.0 Safari/537.36")


# ==================== 基础工具 ====================
def log(msg: str, level: str = "INFO"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


def retry_with_backoff(func, *args, **kwargs):
    FAST_FAIL_KEYS = ("RemoteDisconnected", "Connection aborted",
                      "ReadTimeout", "read timed out", "timed out", "Timeout")
    for i in range(RETRY_COUNT):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            err_str = str(e)
            if any(k in err_str for k in FAST_FAIL_KEYS):
                if i >= 0:
                    log(f"🚫 连接/读超时，直接放弃: {err_str[:60]}", "WARN")
                    return None
            if i < RETRY_COUNT - 1:
                wait = RETRY_DELAY * (2 ** i)
                log(f"⏳ 重试{i+1}/{RETRY_COUNT} 等待{wait}s: {err_str[:80]}", "WARN")
                time.sleep(wait)
    return None


def _dump_debug(df, label: str):
    try:
        os.makedirs("debug", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if isinstance(df, pd.DataFrame):
            df.head(30).to_csv(f"debug/{label}_{ts}.csv", index=False, encoding="utf-8-sig")
            log(f"🐞 落盘: debug/{label}_{ts}.csv | shape={df.shape}")
        else:
            with open(f"debug/{label}_{ts}.txt", "w", encoding="utf-8") as f:
                f.write(f"type={type(df)}\nrepr={repr(df)[:1000]}\n")
    except Exception as e:
        log(f"⚠ 落盘失败: {e}", "WARN")


def _to_num(x):
    try:
        return float(str(x).replace(",", "").replace("%", "").strip())
    except Exception:
        return 0.0


def _to_int(x):
    try:
        return int(float(str(x).replace(",", "").strip()))
    except Exception:
        return 1


def _is_chinese(s: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fa5]", str(s)))


def fmt_mcap(yuan_val):
    y = _to_num(yuan_val)
    if y >= 1e8:
        return f"{y/1e8:.1f}亿"
    elif y >= 1e4:
        return f"{y/1e4:.0f}万"
    return f"{y:.0f}元"


def fmt_net(net_yuan):
    n = _to_num(net_yuan)
    if n == 0:
        return "", False
    if abs(n) >= 1e8:
        return f"{n/1e8:+.2f}亿", n < 0
    return f"{n/1e4:+.0f}万", n < 0


def _find_col(df, *keywords):
    for c in df.columns:
        for kw in keywords:
            if kw in str(c):
                return c
    return None


def parse_ftime(ftime_str):
    if ftime_str is None or (isinstance(ftime_str, float) and np.isnan(ftime_str)):
        return None
    try:
        s = str(int(float(ftime_str))).zfill(6)
        if len(s) >= 4:
            return datetime(2020, 1, 1, int(s[:2]), int(s[2:4]))
    except Exception:
        pass
    try:
        return datetime.strptime(str(ftime_str)[:8], "%H:%M:%S")
    except Exception:
        return None


def is_late_afternoon(ftime_str) -> bool:
    dt = parse_ftime(ftime_str)
    if dt is None:
        return False
    return dt > datetime(dt.year, dt.month, dt.day, 14, 30)


# ==================== 涨停池 ====================
def _get_recent_trade_dates(n: int = 10) -> List[str]:
    try:
        import akshare as ak
        cal = retry_with_backoff(ak.tool_trade_date_hist_sina)
        if cal is None or cal.empty or "trade_date" not in cal.columns:
            return []
        cal = cal.copy()
        cal["trade_date"] = pd.to_datetime(cal["trade_date"])
        cal = cal[cal["trade_date"] <= pd.Timestamp.today()]
        cal = cal.sort_values("trade_date", ascending=False)
        return [d.strftime("%Y%m%d") for d in cal["trade_date"].head(n)]
    except Exception as e:
        log(f"⚠ 交易日历失败: {e}", "WARN")
        return []


def _normalize(df: pd.DataFrame, used_date: str) -> pd.DataFrame:
    mapping = {
        "代码": "code", "名称": "name", "涨跌幅": "pct_change", "最新价": "price",
        "成交额": "amount", "流通市值": "circ_market_cap", "总市值": "total_market_cap",
        "换手率": "turnover", "封板资金": "seal_amount",
        "首次封板时间": "first_time", "最后封板时间": "last_time",
        "炸板次数": "open_times", "连板数": "board_count", "所属行业": "industry",
    }
    for old, new in mapping.items():
        if old in df.columns:
            df = df.rename(columns={old: new})
    for col in ["turnover", "total_market_cap", "circ_market_cap", "board_count",
                "open_times", "amount", "seal_amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col, default in [("board_count", 1), ("turnover", 0.0), ("total_market_cap", 0.0),
                         ("first_time", None), ("open_times", 0), ("industry", "未知"),
                         ("seal_amount", 0.0), ("circ_market_cap", 0.0)]:
        if col not in df.columns:
            df[col] = default
    df.attrs["used_date"] = used_date
    return df


def get_limit_up_pool(trade_date: str = None) -> pd.DataFrame:
    import akshare as ak
    if trade_date:
        candidates = [trade_date]
        log(f"📡 获取涨停池: 指定日期 {trade_date}")
    else:
        dates = _get_recent_trade_dates(10)
        if dates:
            candidates = dates
            log(f"📡 获取涨停池: 交易日历回溯，候选 {candidates[:4]}...")
        else:
            today = datetime.now().strftime("%Y%m%d")
            candidates = [today]
            log(f"📡 获取涨停池: 日历不可用，直接试 today={today}")

    for cand in candidates:
        df = retry_with_backoff(ak.stock_zt_pool_em, cand)
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            log(f"⚠ {cand} 无数据", "WARN")
            continue
        log(f"✅ 涨停池命中 {cand}: {len(df)} 只")
        _dump_debug(df, f"ztpool_ok_{cand}")
        return _normalize(df.copy(), cand)

    log("❌ 最近交易日均无可涨停数据", "ERROR")
    return pd.DataFrame()


# ==================== 新浪数据源工具 ====================
def _sina_symbol(code: str) -> str:
    """600000 → sh600000 / 000001 → sz000001"""
    code = str(code).zfill(6)
    return ("sh" if code.startswith(("6", "9")) else "sz") + code


def _parse_lax_json(raw: str):
    """新浪返回的 JSON 有时 key 无引号（JS 对象），先标准解析，失败则修正 key 再解析"""
    raw = (raw or "").strip()
    if not raw or raw in ("null", "[]", "{}"):
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    fixed = re.sub(r'([{,])\s*([A-Za-z_]\w*)\s*:', r'\1"\2":', raw)
    try:
        return json.loads(fixed)
    except Exception:
        return None


def get_fund_flow_sina(code: str):
    """
    新浪个股资金流趋势（近 20 日）。
    接口：MoneyFlow.ssl_qsfx_zjlrqs
    字段：netamount = 主力净流入（元）
    返回：(net_5, net_10)，单位【元】；失败返回 None。
    """
    import urllib.request
    sym = _sina_symbol(code)
    url = (f"http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
           f"MoneyFlow.ssl_qsfx_zjlrqs?page=1&num=20&sort=opendate&asc=0&daima={sym}")
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": _UA,
            "Referer": "http://vip.stock.finance.sina.com.cn/",
        })
        with urllib.request.urlopen(req, timeout=8) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
    except Exception as e:
        log(f"⚠ 新浪资金流网络失败 {code}: {str(e)[:60]}", "WARN")
        return None

    data = _parse_lax_json(raw)
    if not data or not isinstance(data, list):
        return None

    nets = []
    for row in data[:10]:
        if not isinstance(row, dict):
            continue
        v = row.get("netamount", row.get("net_amount", 0))
        try:
            nets.append(float(str(v).replace(",", "")))
        except Exception:
            continue
    if not nets:
        return None
    return float(sum(nets[:5])), float(sum(nets[:10]))


def get_kline_sina(code: str, days: int = 60):
    """
    新浪日 K 线（不复权）。
    接口：CN_MarketData.getKLineData，scale=240 表示日线。
    返回：DataFrame，列名兼容东财（日期/开盘/最高/最低/收盘/成交量）。
    """
    import urllib.request
    sym = _sina_symbol(code)
    url = (f"http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
           f"CN_MarketData.getKLineData?symbol={sym}&scale=240&ma=no&datalen={days}")
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": _UA,
            "Referer": "http://finance.sina.com.cn/",
        })
        with urllib.request.urlopen(req, timeout=8) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
    except Exception as e:
        log(f"⚠ 新浪K线网络失败 {code}: {str(e)[:60]}", "WARN")
        return None

    data = _parse_lax_json(raw)
    if not data or not isinstance(data, list):
        return None

    rows = []
    for r in data:
        if not isinstance(r, dict):
            continue
        try:
            rows.append({
                "日期": r.get("day"),
                "开盘": float(r.get("open", 0)),
                "最高": float(r.get("high", 0)),
                "最低": float(r.get("low", 0)),
                "收盘": float(r.get("close", 0)),
                "成交量": float(r.get("volume", 0)),
            })
        except Exception:
            continue
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["日期"] = pd.to_datetime(df["日期"])
    df = df.sort_values("日期").reset_index(drop=True)
    return df


# ==================== K线（双源）====================
def get_kline(code: str, days: int = 60):
    """
    K 线双源：东财优先 → 新浪回退。
    东财熔断后自动全部走新浪。
    """
    global _kline_circuit_broken, _kline_fail_count, _kline_fast_fail, _slow_kline_calls
    if _kline_fast_fail:
        return None
    code = str(code).zfill(6)

    # ① 东财（未熔断时尝试）
    if not _kline_circuit_broken:
        t_call = time.time()
        try:
            import akshare as ak
            end = datetime.now()
            start = end - timedelta(days=days * 2)
            df = retry_with_backoff(
                ak.stock_zh_a_hist,
                symbol=code, period="daily",
                start_date=start.strftime("%Y%m%d"),
                end_date=end.strftime("%Y%m%d"),
                adjust="qfq",
            )
            dt = time.time() - t_call
            if dt > SLOW_CALL_THRESHOLD:
                _slow_kline_calls += 1
                log(f"🐢 K线慢调用 {code}: {dt:.1f}s（累计{_slow_kline_calls}/{SLOW_CALL_MAX}）", "WARN")
                if _slow_kline_calls >= SLOW_CALL_MAX:
                    _kline_circuit_broken = True
                    log(f"🚫 东财K线慢调用过多，切换新浪源", "WARN")

            if df is not None and not df.empty:
                if "日期" in df.columns:
                    df["日期"] = pd.to_datetime(df["日期"])
                    df = df.sort_values("日期")
                _kline_fail_count = 0
                return df.tail(days)
        except Exception as e:
            log(f"⚠ 东财K线失败 {code}: {str(e)[:50]}", "WARN")

        _kline_fail_count += 1
        if _kline_fail_count >= KL_INE_FAIL_THRESHOLD:
            _kline_circuit_broken = True
            log(f"🚫 东财K线连续失败{_kline_fail_count}只，切换新浪源", "WARN")

    # ② 新浪（东财熔断或失败后自动走这里）
    df = get_kline_sina(code, days)
    if df is not None and not df.empty:
        _kline_fail_count = 0
        return df

    # ③ 两个源都失败
    _kline_fail_count += 1
    if _kline_fail_count >= KL_INE_FAIL_THRESHOLD:
        _kline_fast_fail = True
        log(f"🚫 东财+新浪K线均失败，完全熔断", "WARN")
    return None


def check_ma_bull(df) -> bool:
    if df is None or len(df) < 22 or "收盘" not in df.columns:
        return False
    try:
        c = df["收盘"]
        ma5 = c.rolling(5).mean().iloc[-1]
        ma10 = c.rolling(10).mean().iloc[-1]
        ma20 = c.rolling(20).mean().iloc[-1]
        return ma5 > ma10 > ma20
    except Exception:
        return False


def check_return_shot(df) -> bool:
    if df is None or len(df) < 10 or "收盘" not in df.columns or "成交量" not in df.columns:
        return False
    try:
        c = df["收盘"]; v = df["成交量"]
        up_yest = (c.iloc[-2] / c.iloc[-3] - 1) >= 0.095
        shrink = v.iloc[-1] < v.iloc[-2]
        ma5 = c.rolling(5).mean().iloc[-1]
        hold_ma = c.iloc[-1] > ma5
        today_chg = abs(c.iloc[-1] / c.iloc[-2] - 1)
        is_small = today_chg < 0.03
        return up_yest and shrink and hold_ma and is_small
    except Exception:
        return False


def check_limit_gene(df, days: int = 20) -> bool:
    if df is None or len(df) < days + 1 or "收盘" not in df.columns:
        return False
    try:
        recent = df.tail(days + 1).iloc[:-1]
        ret = recent["收盘"].pct_change() * 100
        return (ret >= 9.8).any()
    except Exception:
        return False


# ==================== 资金流（双源）====================
def get_fund_flow(code: str):
    """
    资金流双源：东财优先 → 新浪回退。
    东财熔断后自动全部走新浪。
    """
    global _fund_circuit_broken, _fund_fail_count, _fund_fast_fail, _slow_fund_calls
    if _fund_fast_fail:
        return 0.0, 0.0
    code = str(code).zfill(6)

    # ① 东财（未熔断时尝试）
    if not _fund_circuit_broken:
        t_call = time.time()
        try:
            import akshare as ak
            mkt = "sh" if code.startswith(("6", "9")) else "sz"
            df = retry_with_backoff(ak.stock_individual_fund_flow, stock=code, market=mkt)
            dt = time.time() - t_call
            if dt > SLOW_CALL_THRESHOLD:
                _slow_fund_calls += 1
                log(f"🐢 东财资金流慢调用 {code}: {dt:.1f}s（累计{_slow_fund_calls}/{SLOW_CALL_MAX}）", "WARN")
                if _slow_fund_calls >= SLOW_CALL_MAX:
                    _fund_circuit_broken = True
                    log(f"🚫 东财资金流慢调用过多，切换新浪源", "WARN")

            if df is not None and not df.empty:
                col = _find_col(df, "主力净流入")
                if col:
                    df = df.sort_values(df.columns[0], ascending=False)
                    net_5 = pd.to_numeric(df[col].head(5), errors="coerce").sum()
                    net_10 = pd.to_numeric(df[col].head(10), errors="coerce").sum()
                    _fund_fail_count = 0
                    return float(net_5), float(net_10)
        except Exception as e:
            log(f"⚠ 东财资金流失败 {code}: {str(e)[:50]}", "WARN")

        _fund_fail_count += 1
        if _fund_fail_count >= FUND_FAIL_THRESHOLD:
            _fund_circuit_broken = True
            log(f"🚫 东财资金流连续失败{_fund_fail_count}只，切换新浪源", "WARN")

    # ② 新浪（东财熔断或失败后自动走这里）
    r = get_fund_flow_sina(code)
    if r is not None:
        _fund_fail_count = 0
        return r

    # ③ 两个源都失败
    _fund_fail_count += 1
    if _fund_fail_count >= FUND_FAIL_THRESHOLD:
        _fund_fast_fail = True
        log(f"🚫 东财+新浪资金流均失败，完全熔断", "WARN")
    return 0.0, 0.0


# ==================== 板块数据源 ====================
def _fetch_industry_from_em_direct():
    import urllib.request
    url = ("https://push2.eastmoney.com/api/qt/clist/get"
           "?pn=1&pz=200&po=1&np=1&fltt=2&invt=2&fid=f3"
           "&fs=m:90+t:2+f:!50"
           "&fields=f12,f14,f3,f62,f104,f105,f106")
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": _UA,
            "Referer": "https://quote.eastmoney.com/",
            "Accept": "application/json, text/plain, */*",
        })
        with urllib.request.urlopen(req, timeout=8) as resp:
            raw = resp.read().decode("utf-8", errors="ignore").strip()
    except Exception as e:
        log(f"⚠ 东财push2直连失败: {str(e)[:80]}", "WARN")
        return None

    if not raw.startswith("{"):
        i, j = raw.find("{"), raw.rfind("}")
        if i >= 0 and j > i:
            raw = raw[i:j + 1]
    try:
        data = json.loads(raw)
    except Exception as e:
        log(f"⚠ 东财push2 JSON解析失败: {str(e)[:60]}", "WARN")
        return None

    diff = (data.get("data") or {}).get("diff") or []
    if not diff:
        return None

    out = []
    for r in diff:
        name = str(r.get("f14", "")).strip()
        if not name:
            continue
        out.append({
            "板块": name, "5日涨幅": 0.0,
            "当日涨幅": _to_num(r.get("f3", 0)),
            "主力净流入": _to_num(r.get("f62", 0)),
            "来源": "东财push2",
        })
    log(f"✅ 东财push2行业源: {len(out)} 个板块")
    return pd.DataFrame(out)


def _fetch_industry_from_sina():
    """
    新浪行业接口。value 格式："new_blhy,玻璃行业,15,均价,涨跌额,涨跌幅,..."
    扫描前 3 个字段找第一个中文字段作为板块名。
    """
    import urllib.request
    url = "http://vip.stock.finance.sina.com.cn/q/view/newSinaHy.php"
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": _UA,
            "Referer": "http://finance.sina.com.cn/",
        })
        with urllib.request.urlopen(req, timeout=8) as resp:
            raw = resp.read().decode("gbk", errors="ignore")
    except Exception as e:
        log(f"⚠ 新浪行业源失败: {str(e)[:80]}", "WARN")
        return None

    m = re.search(r"=\s*(\{.*\})", raw, re.S)
    if not m:
        log(f"⚠ 新浪行业源格式异常", "WARN")
        return None
    try:
        data = json.loads(m.group(1))
    except Exception as e:
        log(f"⚠ 新浪行业源 JSON 解析失败: {str(e)[:60]}", "WARN")
        return None

    out = []
    skipped = 0
    for code, val in data.items():
        parts = [p.strip() for p in str(val).split(",")]
        if len(parts) < 5:
            skipped += 1
            continue
        name_idx = -1
        for i in range(min(3, len(parts))):
            if _is_chinese(parts[i]):
                name_idx = i
                break
        if name_idx < 0:
            skipped += 1
            continue
        name = parts[name_idx]
        pct_idx = name_idx + 4
        amount_idx = name_idx + 6
        if len(parts) <= pct_idx:
            skipped += 1
            continue
        out.append({
            "板块":     name,
            "5日涨幅":  0.0,
            "当日涨幅": _to_num(parts[pct_idx]),
            "主力净流入": 0.0,
            "成交额万":  _to_num(parts[amount_idx]) if len(parts) > amount_idx else 0.0,
            "来源":     "新浪行业",
        })
    log(f"✅ 新浪行业源: {len(out)} 个板块（跳过 {skipped} 条格式异常）")
    return pd.DataFrame(out)


def _fetch_industry_from_ths():
    import akshare as ak
    df = retry_with_backoff(ak.stock_board_industry_summary_ths)
    if df is None or df.empty:
        return None
    name_col = _find_col(df, "板块名称", "名称", "行业")
    pct_col = _find_col(df, "涨跌幅")
    net_col = _find_col(df, "主力净流入", "净额", "净流入")
    if not name_col:
        return None
    out = []
    for _, r in df.iterrows():
        out.append({"板块": str(r[name_col]),
                    "5日涨幅": _to_num(r.get(pct_col, 0)) if pct_col else 0.0,
                    "当日涨幅": 0.0,
                    "主力净流入": _to_num(r.get(net_col, 0)) if net_col else 0.0,
                    "来源": "同花顺"})
    log(f"✅ 同花顺行业源: {len(out)} 个板块")
    return pd.DataFrame(out)


def _fetch_industry_from_em():
    import akshare as ak
    df = retry_with_backoff(ak.stock_board_industry_name_em)
    if df is None or df.empty:
        return None
    name_col = _find_col(df, "板块名称", "名称")
    pct_col = _find_col(df, "涨跌幅")
    net_col = _find_col(df, "主力净流入", "净额")
    if not name_col:
        return None
    out = []
    for _, r in df.iterrows():
        out.append({"板块": str(r[name_col]),
                    "5日涨幅": _to_num(r.get(pct_col, 0)) if pct_col else 0.0,
                    "当日涨幅": 0.0,
                    "主力净流入": _to_num(r.get(net_col, 0)) if net_col else 0.0,
                    "来源": "东财"})
    log(f"✅ 东财行业源: {len(out)} 个板块")
    return pd.DataFrame(out)


def _fetch_fund_flow_rank():
    import akshare as ak
    try:
        df = retry_with_backoff(ak.stock_sector_fund_flow_rank, "5日", "行业资金流")
    except Exception as e:
        log(f"⚠ 板块资金流排名异常: {e}", "WARN")
        return None
    if df is None or df.empty:
        return None
    name_col = _find_col(df, "板块名称", "板块")
    net_col = _find_col(df, "主力净流入")
    pct_col = _find_col(df, "涨跌幅")
    if not name_col:
        return None
    out = []
    for _, r in df.iterrows():
        out.append({"板块": str(r[name_col]),
                    "5日涨幅": _to_num(r.get(pct_col, 0)) if pct_col else 0.0,
                    "当日涨幅": 0.0,
                    "主力净流入": _to_num(r.get(net_col, 0)) if net_col else 0.0,
                    "来源": "东财资金流"})
    log(f"✅ 东财资金流排名: {len(out)} 个板块")
    return pd.DataFrame(out)


def _build_from_limitup(zt_df):
    if zt_df is None or zt_df.empty or "industry" not in zt_df.columns:
        return pd.DataFrame()

    grp = zt_df.groupby("industry").agg(
        涨停家数=("code", "count"),
        封板资金合计=("seal_amount", "sum"),
        平均封板资金=("seal_amount", "mean"),
        平均换手=("turnover", "mean"),
    ).reset_index()
    grp = grp[~grp["industry"].isin([None, "未知", ""])]
    if grp.empty:
        return pd.DataFrame()

    total_zt = int(grp["涨停家数"].sum())
    strong_threshold = (FALLBACK_STRONG_CNT_LOW if total_zt < FALLBACK_TOTAL_ZT_LOW
                        else FALLBACK_STRONG_CNT)
    log(f"📊 兜底模式：总涨停 {total_zt} 家，强势阈值={strong_threshold} 家")

    def _status(cnt, seal):
        if cnt >= strong_threshold:  return "🔥强势主线"
        elif cnt >= 2:               return "📈升温中"
        elif cnt == 1 and seal >= FALLBACK_WEAK_SEAL: return "🌡️弱势主线"
        return "—"

    out = []
    for _, r in grp.iterrows():
        cnt = int(r["涨停家数"])
        seal = float(r["封板资金合计"] or 0)
        out.append({
            "板块":       str(r["industry"]),
            "5日涨幅":     0.0, "当日涨幅": 0.0, "主力净流入": 0.0,
            "涨停家数":     cnt,
            "封板资金合计":  seal,
            "平均封板资金":  float(r["平均封板资金"] or 0),
            "平均换手":     float(r["平均换手"] or 0),
            "来源":       "涨停池聚合",
            "状态":       _status(cnt, seal),
            "强度分":      cnt * 10,
        })
    df = pd.DataFrame(out).sort_values(
        ["涨停家数", "封板资金合计"], ascending=[False, False]
    ).reset_index(drop=True)
    top3 = " | ".join(f"{r.板块}({r.涨停家数}家)" for r in df.head(3).itertuples())
    log(f"✅ 涨停池聚合主线: {len(df)} 个板块，TOP3: {top3}")
    return df


def get_sector_rotation(zt_df=None):
    rows, seen = [], set()
    status = "ok"
    source_chain = []

    try:
        d = _fetch_industry_from_em_direct()
        if d is not None and not d.empty:
            for _, r in d.iterrows():
                if r["板块"] not in seen:
                    seen.add(r["板块"]); rows.append(r.to_dict())
            source_chain.append("push2")
    except Exception as e:
        log(f"⚠ push2 异常: {e}", "WARN")

    if not rows:
        try:
            d = _fetch_industry_from_sina()
            if d is not None and not d.empty:
                for _, r in d.iterrows():
                    if r["板块"] not in seen:
                        seen.add(r["板块"]); rows.append(r.to_dict())
                source_chain.append("新浪")
        except Exception as e:
            log(f"⚠ 新浪异常: {e}", "WARN")

    if not rows:
        try:
            d = _fetch_industry_from_ths()
            if d is not None and not d.empty:
                for _, r in d.iterrows():
                    if r["板块"] not in seen:
                        seen.add(r["板块"]); rows.append(r.to_dict())
                source_chain.append("同花顺")
        except Exception as e:
            log(f"⚠ 同花顺异常: {e}", "WARN")

    if not rows:
        try:
            d = _fetch_industry_from_em()
            if d is not None and not d.empty:
                for _, r in d.iterrows():
                    if r["板块"] not in seen:
                        seen.add(r["板块"]); rows.append(r.to_dict())
                source_chain.append("东财")
        except Exception as e:
            log(f"⚠ 东财异常: {e}", "WARN")

    try:
        d = _fetch_fund_flow_rank()
        if d is not None and not d.empty:
            for _, r in d.iterrows():
                if r["板块"] in seen:
                    for ex in rows:
                        if ex["板块"] == r["板块"] and r["主力净流入"] != 0:
                            ex["主力净流入"] = r["主力净流入"]
                            if r["5日涨幅"] != 0:
                                ex["5日涨幅"] = r["5日涨幅"]
                            break
                else:
                    seen.add(r["板块"]); rows.append(r.to_dict())
    except Exception as e:
        log(f"⚠ 资金流排名异常: {e}", "WARN")

    if not rows:
        log("⚠ 外部板块源全挂，启用涨停池聚合兜底", "WARN")
        d = _build_from_limitup(zt_df)
        if not d.empty:
            return d, "涨停池聚合兜底（数据置信度：低）"
        return pd.DataFrame(), "全部失败"

    df = pd.DataFrame(rows).drop_duplicates(subset=["板块"]).copy()
    has_5d = "5日涨幅" in df.columns and df["5日涨幅"].abs().sum() > 0
    has_1d = "当日涨幅" in df.columns and df["当日涨幅"].abs().sum() > 0
    has_net = "主力净流入" in df.columns and df["主力净流入"].abs().sum() > 0
    has_amount = "成交额万" in df.columns and df["成交额万"].abs().sum() > 0
    has_real_pct = has_5d or has_1d

    if has_1d:
        pct_median = df["当日涨幅"].median()
    elif has_5d:
        pct_median = df["5日涨幅"].median()
    else:
        pct_median = 0.0
    amt_median = df["成交额万"].median() if has_amount else 0.0

    def _get_pct(r):
        if has_5d and r.get("5日涨幅", 0) != 0:
            return r["5日涨幅"], "5日"
        if has_1d:
            return r.get("当日涨幅", 0), "当日"
        return 0, ""

    def calc_strength(r):
        s = 0
        if has_real_pct:
            pct, kind = _get_pct(r)
            rel = pct - pct_median
            if kind == "当日":
                if pct >= 3:     s += 30
                elif pct >= 1.5: s += 20
                elif pct >= 0.5: s += 15
                elif pct >= 0:   s += 10
                elif pct >= -1:  s += 0
                else:            s -= 10
                if rel >= 1.5:   s += 10
                elif rel >= 0.5: s += 5
            else:
                if pct >= 5:     s += 30
                elif pct >= 2:   s += 20
                elif pct >= 0:   s += 10
                else:            s -= 5
            if has_net:
                net = r["主力净流入"]
                if net > 0:    s += 30
                elif net < 0:  s -= 15
            elif has_amount:
                amt = r.get("成交额万", 0)
                if amt >= amt_median * 1.5:   s += 20
                elif amt >= amt_median:       s += 10
                else:                         s -= 5
            return s
        else:
            cnt = r.get("涨停家数", 0)
            return 50 if cnt >= 5 else 35 if cnt >= 3 else 20 if cnt >= 2 else 10

    df["强度分"] = df.apply(calc_strength, axis=1)

    def label(r):
        s = r["强度分"]
        if not has_real_pct:
            cnt = r.get("涨停家数", 0)
            seal = _to_num(r.get("封板资金合计", 0))
            if cnt >= 3: return "🔥强势主线"
            elif cnt >= 2: return "📈升温中"
            elif cnt == 1 and seal >= FALLBACK_WEAK_SEAL: return "🌡️弱势主线"
            return "—"
        net = r["主力净流入"] if has_net else 0
        pct, _ = _get_pct(r)
        if s >= 60 and (not has_net or net > 0):
            return "🔥强势主线"
        if s >= 40:
            return "📈升温中"
        if WEAK_STRENGTH_LOW <= s < WEAK_STRENGTH_HIGH:
            return "🌡️弱势主线"
        if s < WEAK_STRENGTH_LOW and (not has_net or net < 0):
            return "📉走弱"
        if has_net and pct > 3 and net < 0:
            return "⚠️脉冲"
        return "—"

    df["状态"] = df.apply(label, axis=1)
    sort_cols = ["强度分"]; sort_asc = [False]
    if has_net:
        sort_cols.append("主力净流入"); sort_asc.append(False)
    df = df.sort_values(sort_cols, ascending=sort_asc).reset_index(drop=True)

    cnt_strong = int((df["状态"] == "🔥强势主线").sum())
    cnt_warm   = int((df["状态"] == "📈升温中").sum())
    cnt_weak   = int((df["状态"] == "🌡️弱势主线").sum())
    cnt_down   = int((df["状态"] == "📉走弱").sum())

    calib = "5日" if has_5d else ("当日" if has_1d else "兜底")
    src_str = "+".join(source_chain) if source_chain else "无"
    log(f"✅ 板块主线分析完成: {len(df)} 个板块 | 源={src_str} | 口径={calib}")
    log(f"   标签分布: 🔥{cnt_strong} 📈{cnt_warm} 🌡️{cnt_weak} 📉{cnt_down}")
    return df, status


# ==================== 市场温度 ====================
def calc_market_temperature(zt_df):
    if zt_df is None or zt_df.empty:
        return 0, {"涨停家数": 0, "最高连板": 0, "连板家数": 0, "炸板率": "N/A"}

    zt_count = len(zt_df)

    if zt_count >= 80:   cnt_score = 40
    elif zt_count >= 60: cnt_score = 35
    elif zt_count >= 40: cnt_score = 25
    elif zt_count >= 20: cnt_score = 15
    else:                cnt_score = 5

    max_board = int(zt_df["board_count"].max()) if "board_count" in zt_df.columns else 1
    multi_board = int((zt_df["board_count"] >= 2).sum()) if "board_count" in zt_df.columns else 0
    ladder_score = 0
    if max_board >= 4:      ladder_score += 15
    elif max_board >= 3:    ladder_score += 10
    elif max_board >= 2:    ladder_score += 5
    if multi_board >= 10:   ladder_score += 15
    elif multi_board >= 5:  ladder_score += 10
    elif multi_board >= 2:  ladder_score += 5

    if "open_times" in zt_df.columns and zt_count > 0:
        broken = int((zt_df["open_times"] > 0).sum())
        broken_ratio = broken / zt_count
    else:
        broken_ratio = 0.3
    if broken_ratio <= 0.15:   br_score = 30
    elif broken_ratio <= 0.25: br_score = 20
    elif broken_ratio <= 0.35: br_score = 10
    else:                      br_score = 0

    temp = cnt_score + ladder_score + br_score
    detail = {
        "涨停家数": zt_count,
        "最高连板": max_board,
        "连板家数": multi_board,
        "炸板率":   f"{broken_ratio:.1%}",
    }
    return temp, detail


def temp_label(temp):
    if temp >= TEMP_HOT:   return "🔥 热", "满仓操作，主升浪可期"
    if temp >= TEMP_WARM:  return "☀️ 温", "半仓操作，优先主线龙头"
    if temp >= TEMP_COLD:  return "🌥️ 冷", "轻仓试探，严格止损"
    return "❄️ 冰冻", "建议空仓观望"


# ==================== 硬过滤 ====================
def hard_filter(df):
    if df.empty:
        return df, []
    original = len(df)
    reasons = []

    if HARD_FILTERS["st"] and "name" in df.columns:
        mask = df["name"].astype(str).str.contains("ST|退", case=False, na=False)
        if mask.any():
            reasons.append(f"ST/退市: {mask.sum()}只"); df = df[~mask]

    if "board_count" in df.columns:
        mask = df["board_count"] > HARD_FILTERS["max_boards"]
        if mask.any():
            reasons.append(f"连板>3: {mask.sum()}只"); df = df[~mask]

    if "turnover" in df.columns:
        mask = df["turnover"] > HARD_FILTERS["max_turnover"]
        if mask.any():
            reasons.append(f"换手>28%: {mask.sum()}只"); df = df[~mask]

    if HARD_FILTERS["one_word"] and "seal_amount" in df.columns and "circ_market_cap" in df.columns:
        circ = df["circ_market_cap"].replace(0, np.nan)
        ratio = df["seal_amount"] / circ
        mask = (df["open_times"].fillna(99) == 0) & (ratio > HARD_FILTERS["seal_ratio_one_word"])
        if mask.any():
            reasons.append(f"一字板: {mask.sum()}只"); df = df[~mask]

    if "total_market_cap" in df.columns:
        min_cap_yuan = HARD_FILTERS["min_market_cap"] * 1e8
        mask = df["total_market_cap"] < min_cap_yuan
        if mask.any():
            reasons.append(f"市值<{HARD_FILTERS['min_market_cap']}亿: {mask.sum()}只")
            df = df[~mask]

    if HARD_FILTERS["late_afternoon"] and "first_time" in df.columns:
        mask = df["first_time"].apply(lambda x: is_late_afternoon(x))
        if mask.any():
            reasons.append(f"尾盘偷袭: {mask.sum()}只"); df = df[~mask]

    log(f"🔍 硬过滤: 剔除{original - len(df)}只 ({', '.join(reasons) or '无'})")
    return df, reasons


# ==================== 评分 ====================
def score_stock(row, sector_counts, use_fund, use_ma):
    """
    返回 (原始分, 满分, 加分理由)
    满分口径：
      - board_count 最多 15
      - fund_flow 两项只在 use_fund 时计入
      - ma_bull / return_shot / limit_gene 只在 use_ma 时计入
    """
    global _fund_circuit_broken, _kline_circuit_broken, _fund_fast_fail, _kline_fast_fail
    weights = WEIGHTS_DEGRADED if (_fund_fast_fail or _kline_fast_fail) else WEIGHTS_NORMAL
    score = 0
    max_score = 0
    reasons = []

    # 1) 连板
    max_score += 15
    bc = _to_int(row.get("board_count", 1))
    if bc >= 2:
        s = min(bc * weights["board_count"], 15)
        score += s
        reasons.append(f"连板{bc}层(+{s})")

    # 2) 主线
    max_score += weights["sector_main"]
    ind = str(row.get("industry", "未知"))
    if sector_counts.get(ind, 0) >= 3:
        s = weights["sector_main"]
        score += s
        reasons.append(f"主线({ind}{sector_counts[ind]}家)(+{s})")

    # 3) 资金流
    if use_fund and weights["fund_flow_5d"] > 0:
        max_score += weights["fund_flow_5d"] + weights["fund_flow_10d"]
        n5, n10 = get_fund_flow(str(row.get("code", "")).zfill(6))
        if n5 > 0:
            s = min(int(n5 / 1000), weights["fund_flow_5d"]); score += s
            reasons.append(f"5日+{n5:.0f}万(+{s})")
        if n10 > 0:
            s = min(int(n10 / 2000), weights["fund_flow_10d"]); score += s
            reasons.append(f"10日+{n10:.0f}万(+{s})")

    # 4) 早盘封板
    max_score += weights["morning_lobby"]
    ft = parse_ftime(row.get("first_time"))
    if ft and ft.hour < 10:
        s = weights["morning_lobby"]; score += s
        reasons.append(f"早盘{ft.strftime('%H:%M')}(+{s})")

    # 5) 未开板
    max_score += weights["no_open"]
    if _to_int(row.get("open_times", 1)) == 0:
        s = weights["no_open"]; score += s
        reasons.append(f"未开板(+{s})")

    # 6) 均线 / 回马枪 / 涨停基因
    if use_ma and weights["ma_bull"] > 0:
        max_score += weights["ma_bull"] + weights["return_shot"]
        if weights["limit_gene"] > 0:
            max_score += weights["limit_gene"]
        kl = get_kline(str(row.get("code", "")).zfill(6))
        if check_ma_bull(kl):
            s = weights["ma_bull"]; score += s
            reasons.append(f"均线多头(+{s})")
        if check_return_shot(kl):
            s = weights["return_shot"]; score += s
            reasons.append(f"回马枪(+{s})")
        if weights["limit_gene"] > 0 and check_limit_gene(kl, 20):
            s = weights["limit_gene"]; score += s
            reasons.append(f"20日涨停基因(+{s})")

    # 7) 市值加分
    max_score += weights["market_cap_bonus"]
    mcap = _to_num(row.get("total_market_cap", 0))
    if 30 <= mcap / 1e8 <= 100:
        s = weights["market_cap_bonus"]; score += s
        reasons.append(f"市值{fmt_mcap(mcap)}(+{s})")

    # 8) 换手健康
    max_score += weights["turnover_good"]
    to = _to_num(row.get("turnover", 0))
    if 5 <= to <= 15:
        s = weights["turnover_good"]; score += s
        reasons.append(f"换手{to:.1f}%健康(+{s})")

    # 9) 封单
    max_score += weights["seal_amount"]
    if _to_num(row.get("seal_amount", 0)) > 0:
        s = weights["seal_amount"]; score += s
        reasons.append(f"有封单(+{s})")

    return score, max_score, reasons


# ==================== 推送 ====================
def send_wecom_webhook(webhook_url, content):
    if not webhook_url:
        log("⚠ WECOM_WEBHOOK 未配置", "WARN")
        return False
    try:
        import urllib.request
        req = urllib.request.Request(
            webhook_url,
            data=json.dumps({"msgtype": "markdown", "markdown": {"content": content}}).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            r = json.loads(resp.read().decode("utf-8"))
            if r.get("errcode") == 0:
                log("✅ 企微推送成功"); return True
            log(f"❌ 推送失败: {r}", "ERROR")
    except Exception as e:
        log(f"❌ 推送异常: {e}", "ERROR")
    return False


def _fmt_sector_line(r):
    p5 = _to_num(r.get("5日涨幅", 0))
    p1 = _to_num(r.get("当日涨幅", 0))
    net = _to_num(r.get("主力净流入", 0))
    cnt = _to_int(r.get("涨停家数", 0))

    if p5 != 0:
        pct_str = f"5日{p5:+.1f}%"
    elif p1 != 0:
        pct_str = f"当日{p1:+.1f}%"
    else:
        pct_str = ""

    net_str, is_out = fmt_net(net)
    if net_str and is_out:
        net_str = "资金" + net_str + " ↘"

    if cnt > 0 and not pct_str:
        seal = _to_num(r.get("封板资金合计", 0))
        seal_str = f" | 封板合计{seal/1e8:.2f}亿" if seal > 0 else ""
        return f"- {r['板块']} | 涨停{cnt}家{seal_str}（兜底口径）"

    parts = [r['板块']]
    if pct_str: parts.append(pct_str)
    if net_str: parts.append(net_str)
    return f"- {' | '.join(parts)}"


def format_market_temp_section(temp, detail):
    label, advice = temp_label(temp)
    msg = "\n### 🌡️ 市场温度\n"
    msg += f"**{label}：{temp}/100**\n"
    msg += f"> 涨停 {detail['涨停家数']} 家 | 最高连板 {detail['最高连板']} 层 | "
    msg += f"连板家数 {detail['连板家数']} 家 | 炸板率 {detail['炸板率']}\n"
    msg += f"> **建议**：{advice}\n"
    return msg


def format_sector_section(sec_df, status):
    if sec_df is None or sec_df.empty:
        return "\n### 🌐 板块主线监测\n> ⚠️ 本次未取到任何板块数据，主线判断暂缺\n"

    is_fallback = sec_df.attrs.get("is_fallback", False) or (status and "兜底" in str(status))
    msg = "\n### 🌐 板块主线监测\n"
    if status and status != "ok":
        msg += f"> 状态: {status}\n"
    if is_fallback:
        msg += "> ⚠️ 数据置信度：低（无板块涨幅/资金流，仅按涨停家数聚合）\n"

    strong = sec_df[sec_df["状态"] == "🔥强势主线"].head(6)
    if not strong.empty:
        msg += "\n**🔥 强势主线**\n"
        for _, r in strong.iterrows():
            msg += _fmt_sector_line(r) + "\n"
    else:
        msg += "\n> 🔥 暂无达到阈值的强势主线\n"

    weak = sec_df[sec_df["状态"] == "🌡️弱势主线"].head(3)
    if not weak.empty:
        msg += "\n**🌡️ 衰退预警**（涨幅微弱、资金不足，可能是退潮主线）\n"
        for _, r in weak.iterrows():
            msg += _fmt_sector_line(r) + "\n"

    return msg


def format_message(df, mode, tag, filtered_reasons, sec_df, sec_status,
                   min_score_used, is_degraded, temp, temp_detail):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    used = df.attrs.get("used_date", "") if hasattr(df, "attrs") else ""
    used_line = f" | 数据日:{used}" if used else ""
    mode_str = "降级模式" if is_degraded else "正常"

    msg = f"## 📈 每日选股推送 - {tag}\n"
    msg += f"**时间**: {now}{used_line} | **模式**: {mode}\n"
    msg += f"**阈值**: {min_score_used}分（{mode_str}）\n"

    msg += format_market_temp_section(temp, temp_detail)
    msg += format_sector_section(sec_df, sec_status)

    msg += f"\n### 🏆 候选标的（≥{min_score_used}分，{len(df)}只）\n"

    if temp < TEMP_COLD:
        msg += "> ❄️ **市场温度过低，建议空仓**。以下候选仅供参考，不建议实际买入。\n"
    elif df.empty:
        msg += "> 今日无符合标准的标的，建议空仓观望。\n"

    if not df.empty:
        for i, (_, r) in enumerate(df.head(15).iterrows(), 1):
            msg += f"\n**{i}. {r.get('name','')}** (`{r.get('code','')}`)\n"
            raw = r.get("raw_score", 0)
            maxs = r.get("max_score", 100)
            msg += f"- 评分:**{r.get('score',0)}/100**（原始 {raw}/{maxs}）| 板块:{r.get('industry','N/A')} | 连板:{_to_int(r.get('board_count',1))}\n"
            msg += f"- 换手:{_to_num(r.get('turnover',0)):.1f}% | 市值:{fmt_mcap(r.get('total_market_cap',0))}\n"
            if "reasons" in r and r["reasons"]:
                msg += f"- 亮点: {'、'.join(r['reasons'][:3])}\n"

    msg += "\n### 📊 过滤统计\n"
    for r in filtered_reasons:
        msg += f"- {r}\n"

    msg += "\n> 💡 初筛结果，不构成投资建议。"
    return msg


# ==================== 主流程 ====================
def main():
    global _fund_circuit_broken, _kline_circuit_broken
    global _fund_fail_count, _kline_fail_count, _fund_fast_fail, _kline_fast_fail

    p = argparse.ArgumentParser(description="每日选股推送")
    p.add_argument("--no-ma", action="store_true")
    p.add_argument("--no-fund", action="store_true")
    p.add_argument("--top", type=int, default=DEFAULT_TOP_N)
    p.add_argument("--date", type=str, default="")
    p.add_argument("--min-score", type=int, default=None)
    args = p.parse_args()

    if args.no_ma and args.no_fund:
        mode = "fast"
    elif args.no_fund:
        mode = "no_fund"
    elif args.no_ma:
        mode = "no_ma"
    else:
        mode = "full"

    tag = f"run-{os.environ.get('GITHUB_RUN_ID', 'local')}"
    log(f"🚀 启动 v6.3 | 模式={mode} | 日期={args.date or '今日(回溯)'}")

    try:
        user_specified_threshold = args.min_score is not None
        min_score = args.min_score if user_specified_threshold else DEFAULT_MIN_SCORE

        # 1. 涨停池
        df_raw = get_limit_up_pool(args.date)
        if df_raw.empty:
            send_wecom_webhook(os.environ.get("WECOM_WEBHOOK", ""),
                "## ⚠️ 选股未执行\n**时间**: " + datetime.now().strftime("%Y-%m-%d %H:%M") +
                "\n**原因**: 最近交易日涨停池均为空")
            sys.exit(1)

        # 2. 市场温度
        temp, temp_detail = calc_market_temperature(df_raw)
        temp_lbl, temp_advice = temp_label(temp)
        log(f"🌡️ 市场温度: {temp}/100 ({temp_lbl}) | 涨停{temp_detail['涨停家数']}家 "
            f"最高{temp_detail['最高连板']}板 连板{temp_detail['连板家数']}家 炸板率{temp_detail['炸板率']}")
        log(f"   建议: {temp_advice}")

        # 3. 板块主线
        log("🌐 板块主线监测...")
        sec_df, sec_status = get_sector_rotation(df_raw)
        if not sec_df.empty:
            sec_df.attrs["is_fallback"] = "兜底" in str(sec_status)
        log(f"🌐 板块数据 rows={len(sec_df)} | status={sec_status}")
        if not sec_df.empty:
            top3_parts = [f"{r['板块']}({r.get('状态','—')})" for _, r in sec_df.head(3).iterrows()]
            log(f"✅ 板块TOP3: {' | '.join(top3_parts)}")
            _dump_debug(sec_df, "sector_rotation_result")

        # 4. 板块统计
        sector_counts = {}
        if "industry" in df_raw.columns:
            sector_counts = df_raw["industry"].value_counts().to_dict()
            top3 = sorted(sector_counts.items(), key=lambda x: (-x[1], x[0]))[:3]
            log(f"📊 涨停板块TOP3: {' | '.join(f'{k}({v})' for k,v in top3)}")

        # 5. 硬过滤
        df, filtered_reasons = hard_filter(df_raw)
        if df.empty:
            log("📭 过滤后无标的")
            send_wecom_webhook(os.environ.get("WECOM_WEBHOOK", ""),
                              format_message(df, mode, tag, filtered_reasons, sec_df, sec_status,
                                             min_score, is_degraded=False, temp=temp, temp_detail=temp_detail))
            return

        # 6. 探针预检（v6.3：探针成功即继续）
        log(f"🔬 探针预检接口可用性（{PROBE_N} 只）...")
        probe_fund_ok = False
        probe_kline_ok = False
        probe = df.head(min(PROBE_N, len(df)))
        for _, row in probe.iterrows():
            code = str(row.get("code", "")).zfill(6)
            if not args.no_fund:
                n5, n10 = get_fund_flow(code)
                if n5 != 0 or n10 != 0:
                    probe_fund_ok = True
            if not args.no_ma:
                kl = get_kline(code)
                if kl is not None and not kl.empty:
                    probe_kline_ok = True

        # 探针失败：完全熔断对应接口
        if not args.no_fund and not probe_fund_ok:
            _fund_fast_fail = True
            log(f"🚫 探针资金流：东财+新浪均失败，完全熔断", "WARN")
        if not args.no_ma and not probe_kline_ok:
            _kline_fast_fail = True
            log(f"🚫 探针K线：东财+新浪均失败，完全熔断", "WARN")

        log(f"🔬 预检完成: 资金={('✅可用' if probe_fund_ok else '❌熔断')} "
            f"K线={('✅可用' if probe_kline_ok else '❌熔断')}")

        # 7. 降级判定
        use_fund = not args.no_fund and not _fund_fast_fail
        use_ma = not args.no_ma and not _kline_fast_fail
        is_degraded = False
        if (_fund_fast_fail or _kline_fast_fail) and not user_specified_threshold:
            min_score = DEGRADED_MIN_SCORE
            is_degraded = True
            log(f"⚠️ 接口降级，阈值自动从{DEFAULT_MIN_SCORE}调整为{min_score}", "WARN")

        # 8. 评分（归一化）
        log(f"📝 开始评分 {len(df)} 只（资金={'开' if use_fund else '关'} "
            f"均线={'开' if use_ma else '关'} 阈值={min_score} 预算={SCORE_TIME_BUDGET}s）...")
        t_start = time.time()
        budget_exhausted = False
        raw_scores, max_scores, norm_scores, all_reasons = [], [], [], []

        for idx, (_, row) in enumerate(df.iterrows()):
            elapsed = time.time() - t_start
            if elapsed > SCORE_TIME_BUDGET and not budget_exhausted:
                budget_exhausted = True
                if not _fund_fast_fail:
                    _fund_fast_fail = True
                if not _kline_fast_fail:
                    _kline_fast_fail = True
                log(f"⏰ 评分阶段超预算 {elapsed:.0f}s，剩余 {len(df)-idx} 只改走降级权重", "WARN")
                if not user_specified_threshold:
                    min_score = DEGRADED_MIN_SCORE
                    is_degraded = True

            _use_net = not budget_exhausted and not (_fund_fast_fail and _kline_fast_fail)
            raw, mx, rs = score_stock(row, sector_counts,
                                      use_fund=use_fund and _use_net and not _fund_fast_fail,
                                      use_ma=use_ma and _use_net and not _kline_fast_fail)
            norm = int(round(raw / mx * 100)) if mx > 0 else 0
            raw_scores.append(raw)
            max_scores.append(mx)
            norm_scores.append(norm)
            all_reasons.append(rs)

            if (idx + 1) % 10 == 0:
                log(f"  ⏳ {idx+1}/{len(df)} 只 ({time.time()-t_start:.1f}s)")

        df = df.copy()
        df["raw_score"] = raw_scores
        df["max_score"] = max_scores
        df["score"] = norm_scores
        df["reasons"] = all_reasons
        df_f = df[df["score"] >= min_score].sort_values(
            ["score", "raw_score"], ascending=[False, False]
        ).head(args.top)
        log(f"✅ 达标 {len(df_f)} 只 ≥ {min_score}分 ({time.time()-t_start:.1f}s)")

        # 9. 落盘
        out_dir = os.environ.get("OUTPUT_DIR", "results")
        os.makedirs(out_dir, exist_ok=True)
        if not df_f.empty:
            cols = [c for c in ["code", "name", "score", "raw_score", "max_score",
                                 "industry", "board_count", "turnover", "total_market_cap"]
                    if c in df_f.columns]
            df_f[cols].to_csv(
                f"{out_dir}/pick_{datetime.now().strftime('%Y%m%d')}_{mode}.csv",
                index=False, encoding="utf-8-sig")
            log(f"💾 已保存 {len(df_f)}只")

        # 10. 推送
        send_wecom_webhook(os.environ.get("WECOM_WEBHOOK", ""),
                          format_message(df_f, mode, tag, filtered_reasons, sec_df, sec_status,
                                         min_score, is_degraded, temp, temp_detail))
        log("✅ 选股完成")

    except Exception as e:
        log(f"❌ 异常: {e}", "ERROR")
        traceback.print_exc()
        send_wecom_webhook(os.environ.get("WECOM_WEBHOOK", ""),
            f"## ❌ 脚本异常\n**时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n**错误**: {str(e)[:300]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
