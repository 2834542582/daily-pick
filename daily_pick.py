#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每日选股推送 v7.2 —— 早晚双推 + 三段式主线
===========================================================
v7.2 相对 v7.1 的改动：
  【修复】
  1) 板块主线三段式：🔥强势 + 📈升温 + 📉走弱
  2) 早盘提醒节后降级（不直接跳过，改为精简提醒）
  3) hits 里资金显示具体金额（不再只显示"+"）
  4) 主线监控标注数据源（新浪/东财，是否有资金流）
  5) 企微消息超长检查 + 截断（防静默丢弃）
  6) 数据日 != 今天时日志告警
  7) brief 模式读 CSV 加 dtype={"code": str}（防前导0丢失）
  8) brief 模式空候选分支

v7.1 保留：
  - 双源架构（东财优先 → 新浪回退）
  - 市场温度（涨停+梯队+炸板率）
  - 交易日过滤（定时触发时跳过非交易日）
  - 探针即熔断东财
  - --brief 模式（读历史 CSV）

推送时间表：
  - 北京 16:00 工作日 → 主推送
  - 北京 08:50 工作日次日 → 早盘提醒
  - 北京 09:00 周日 → 保活
===========================================================
"""
# ==================== requests 超时补丁 ====================
import requests

_ORIG = requests.Session.request
_TIMEOUT = (3, 8)


def _patched(self, method, url, **kw):
    if kw.get("timeout") is None:
        kw["timeout"] = _TIMEOUT
    return _ORIG(self, method, url, **kw)


requests.Session.request = _patched
_ORIG_API = requests.api.request


def _patched_api(method, url, **kw):
    if kw.get("timeout") is None:
        kw["timeout"] = _TIMEOUT
    return _ORIG_API(method, url, **kw)


requests.api.request = _patched_api
requests.get = lambda url, **kw: _patched_api("get", url, **kw)
requests.post = lambda url, **kw: _patched_api("post", url, **kw)
# ============================================================

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
KLINE_FAIL_THRESHOLD = 3
DEFAULT_MIN_SCORE = 75
DEGRADED_MIN_SCORE = 55

SCORE_TIME_BUDGET = 120
SLOW_CALL_THRESHOLD = 3.0
SLOW_CALL_MAX = 5

WEAK_STRENGTH_LOW = 10
WEAK_STRENGTH_HIGH = 40
STRONG_THRESHOLD_WITH_NET = 60
STRONG_THRESHOLD_NO_NET = 50

TEMP_HOT = 75
TEMP_WARM = 55
TEMP_COLD = 35

BRIEF_MAX_DAYS = 15      # v7.2: 5 → 15，允许跨长假
WECOM_MAX_BYTES = 4000   # 企微 markdown 上限约 4096 字节，留余量

HARD_FILTERS = {
    "st": True, "max_boards": 3, "max_turnover": 28.0,
    "one_word": True, "min_market_cap": 15.0, "late_afternoon": True,
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

# ==================== 全局状态 ====================
_fund_circuit_broken = False
_fund_fail_count = 0
_fund_em_fail = 0
_fund_fast_fail = False
_kline_circuit_broken = False
_kline_fail_count = 0
_kline_em_fail = 0
_kline_fast_fail = False
_slow_fund_calls = 0
_slow_kline_calls = 0
_trade_calendar_cache = None

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
       "AppleWebKit/537.36 (KHTML, like Gecko) "
       "Chrome/120.0.0.0 Safari/537.36")


# ==================== 基础工具 ====================
def log(msg, level="INFO"):
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


def _dump_debug(df, label):
    try:
        os.makedirs("debug", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if isinstance(df, pd.DataFrame):
            df.head(50).to_csv(f"debug/{label}_{ts}.csv", index=False, encoding="utf-8-sig")
    except Exception:
        pass


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


def _is_chinese(s):
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


def fmt_money_short(yuan_val):
    """简短金额显示：+1234万 / +1.5亿"""
    n = _to_num(yuan_val)
    if n == 0:
        return ""
    if abs(n) >= 1e8:
        return f"{n/1e8:+.1f}亿"
    return f"{n/1e4:+.0f}万"


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


def is_late_afternoon(ftime_str):
    dt = parse_ftime(ftime_str)
    if dt is None:
        return False
    return dt > datetime(dt.year, dt.month, dt.day, 14, 30)


# ==================== 交易日历 ====================
def _get_trade_calendar():
    global _trade_calendar_cache
    if _trade_calendar_cache is not None:
        return _trade_calendar_cache
    try:
        import akshare as ak
        cal = retry_with_backoff(ak.tool_trade_date_hist_sina)
        if cal is not None and not cal.empty and "trade_date" in cal.columns:
            _trade_calendar_cache = set(
                pd.to_datetime(cal["trade_date"]).dt.strftime("%Y%m%d")
            )
            return _trade_calendar_cache
    except Exception:
        pass
    return None


def is_trade_date(date_str):
    cal = _get_trade_calendar()
    if cal:
        return date_str in cal
    try:
        d = datetime.strptime(date_str, "%Y%m%d")
        return d.weekday() < 5
    except Exception:
        return False


def calc_next_trade_date(used_date_str):
    if not used_date_str or len(used_date_str) != 8:
        return "下一交易日"
    try:
        d = datetime.strptime(used_date_str, "%Y%m%d")
    except Exception:
        return "下一交易日"

    cal = _get_trade_calendar()
    if cal:
        for i in range(1, 15):
            nxt = d + timedelta(days=i)
            if nxt.strftime("%Y%m%d") in cal:
                return nxt.strftime("%Y-%m-%d")

    nxt = d + timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += timedelta(days=1)
    return nxt.strftime("%Y-%m-%d")


def _get_recent_trade_dates(n=10):
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
    except Exception:
        return []


# ==================== 涨停池 ====================
def _normalize(df, used_date):
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


def get_limit_up_pool(trade_date=None):
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

    for cand in candidates:
        df = retry_with_backoff(ak.stock_zt_pool_em, cand)
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            continue
        log(f"✅ 涨停池命中 {cand}: {len(df)} 只")
        _dump_debug(df, f"ztpool_ok_{cand}")
        return _normalize(df.copy(), cand)

    log("❌ 最近交易日均无可涨停数据", "ERROR")
    return pd.DataFrame()


# ==================== 新浪数据源 ====================
def _sina_symbol(code):
    code = str(code).zfill(6)
    return ("sh" if code.startswith(("6", "9")) else "sz") + code


def _parse_lax_json(raw):
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


def get_fund_flow_sina(code):
    import urllib.request
    sym = _sina_symbol(code)
    url = (f"http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
           f"MoneyFlow.ssl_qsfx_zjlrqs?page=1&num=20&sort=opendate&asc=0&daima={sym}")
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": _UA, "Referer": "http://vip.stock.finance.sina.com.cn/"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
    except Exception:
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


def get_kline_sina(code, days=60):
    import urllib.request
    sym = _sina_symbol(code)
    url = (f"http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
           f"CN_MarketData.getKLineData?symbol={sym}&scale=240&ma=no&datalen={days}")
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": _UA, "Referer": "http://finance.sina.com.cn/"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
    except Exception:
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
    return df.sort_values("日期").reset_index(drop=True)


# ==================== K线（双源）====================
def get_kline(code, days=60):
    global _kline_circuit_broken, _kline_fail_count, _kline_fast_fail, _kline_em_fail, _slow_kline_calls
    if _kline_fast_fail:
        return None
    code = str(code).zfill(6)

    if not _kline_circuit_broken:
        t0 = time.time()
        try:
            import akshare as ak
            end = datetime.now()
            start = end - timedelta(days=days * 2)
            df = retry_with_backoff(
                ak.stock_zh_a_hist, symbol=code, period="daily",
                start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"),
                adjust="qfq")
            dt = time.time() - t0
            if dt > SLOW_CALL_THRESHOLD:
                _slow_kline_calls += 1
                if _slow_kline_calls >= SLOW_CALL_MAX:
                    _kline_circuit_broken = True
            if df is not None and not df.empty:
                if "日期" in df.columns:
                    df["日期"] = pd.to_datetime(df["日期"])
                    df = df.sort_values("日期")
                _kline_fail_count = 0
                _kline_em_fail = 0
                return df.tail(days)
        except Exception:
            pass
        _kline_em_fail += 1
        if _kline_em_fail >= KLINE_FAIL_THRESHOLD:
            _kline_circuit_broken = True

    df = get_kline_sina(code, days)
    if df is not None and not df.empty:
        _kline_fail_count = 0
        return df

    _kline_fail_count += 1
    if _kline_fail_count >= KLINE_FAIL_THRESHOLD:
        _kline_fast_fail = True
    return None


def check_ma_bull(df):
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


def check_return_shot(df):
    if df is None or len(df) < 10 or "收盘" not in df.columns or "成交量" not in df.columns:
        return False
    try:
        c = df["收盘"]; v = df["成交量"]
        up_yest = (c.iloc[-2] / c.iloc[-3] - 1) >= 0.095
        shrink = v.iloc[-1] < v.iloc[-2]
        ma5 = c.rolling(5).mean().iloc[-1]
        hold_ma = c.iloc[-1] > ma5
        is_small = abs(c.iloc[-1] / c.iloc[-2] - 1) < 0.03
        return up_yest and shrink and hold_ma and is_small
    except Exception:
        return False


def check_limit_gene(df, days=20):
    if df is None or len(df) < days + 1 or "收盘" not in df.columns:
        return False
    try:
        recent = df.tail(days + 1).iloc[:-1]
        ret = recent["收盘"].pct_change() * 100
        return (ret >= 9.8).any()
    except Exception:
        return False


# ==================== 资金流（双源）====================
def get_fund_flow(code):
    global _fund_circuit_broken, _fund_fail_count, _fund_fast_fail, _fund_em_fail, _slow_fund_calls
    if _fund_fast_fail:
        return 0.0, 0.0
    code = str(code).zfill(6)

    if not _fund_circuit_broken:
        t0 = time.time()
        try:
            import akshare as ak
            mkt = "sh" if code.startswith(("6", "9")) else "sz"
            df = retry_with_backoff(ak.stock_individual_fund_flow, stock=code, market=mkt)
            dt = time.time() - t0
            if dt > SLOW_CALL_THRESHOLD:
                _slow_fund_calls += 1
                if _slow_fund_calls >= SLOW_CALL_MAX:
                    _fund_circuit_broken = True
            if df is not None and not df.empty:
                col = _find_col(df, "主力净流入")
                if col:
                    df = df.sort_values(df.columns[0], ascending=False)
                    n5 = pd.to_numeric(df[col].head(5), errors="coerce").sum()
                    n10 = pd.to_numeric(df[col].head(10), errors="coerce").sum()
                    _fund_fail_count = 0
                    _fund_em_fail = 0
                    return float(n5), float(n10)
        except Exception:
            pass
        _fund_em_fail += 1
        if _fund_em_fail >= FUND_FAIL_THRESHOLD:
            _fund_circuit_broken = True

    r = get_fund_flow_sina(code)
    if r is not None:
        _fund_fail_count = 0
        return r

    _fund_fail_count += 1
    if _fund_fail_count >= FUND_FAIL_THRESHOLD:
        _fund_fast_fail = True
    return 0.0, 0.0


# ==================== 板块数据 ====================
def _fetch_industry_from_em_direct():
    import urllib.request
    url = ("https://push2.eastmoney.com/api/qt/clist/get"
           "?pn=1&pz=200&po=1&np=1&fltt=2&invt=2&fid=f3"
           "&fs=m:90+t:2+f:!50&fields=f12,f14,f3,f62,f104,f105,f106")
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": _UA, "Referer": "https://quote.eastmoney.com/",
            "Accept": "application/json, text/plain, */*"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            raw = resp.read().decode("utf-8", errors="ignore").strip()
    except Exception as e:
        log(f"⚠ 东财push2失败: {str(e)[:60]}", "WARN")
        return None

    if not raw.startswith("{"):
        i, j = raw.find("{"), raw.rfind("}")
        if i >= 0 and j > i:
            raw = raw[i:j + 1]
    try:
        data = json.loads(raw)
    except Exception:
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
            "来源": "push2",
        })
    log(f"✅ 东财push2: {len(out)} 个板块")
    return pd.DataFrame(out)


def _fetch_industry_from_sina():
    import urllib.request
    url = "http://vip.stock.finance.sina.com.cn/q/view/newSinaHy.php"
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": _UA, "Referer": "http://finance.sina.com.cn/"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            raw = resp.read().decode("gbk", errors="ignore")
    except Exception as e:
        log(f"⚠ 新浪行业失败: {str(e)[:60]}", "WARN")
        return None

    m = re.search(r"=\s*(\{.*\})", raw, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(1))
    except Exception:
        return None

    out = []
    for code, val in data.items():
        parts = [p.strip() for p in str(val).split(",")]
        if len(parts) < 5:
            continue
        name_idx = -1
        for i in range(min(3, len(parts))):
            if _is_chinese(parts[i]):
                name_idx = i
                break
        if name_idx < 0:
            continue
        pct_idx = name_idx + 4
        amt_idx = name_idx + 6
        if len(parts) <= pct_idx:
            continue
        out.append({
            "板块": parts[name_idx], "5日涨幅": 0.0,
            "当日涨幅": _to_num(parts[pct_idx]),
            "主力净流入": 0.0,
            "成交额万": _to_num(parts[amt_idx]) if len(parts) > amt_idx else 0.0,
            "来源": "新浪",
        })
    log(f"✅ 新浪行业: {len(out)} 个板块")
    return pd.DataFrame(out)


def _build_from_limitup(zt_df):
    if zt_df is None or zt_df.empty or "industry" not in zt_df.columns:
        return pd.DataFrame()
    grp = zt_df.groupby("industry").agg(
        涨停家数=("code", "count"),
        封板资金合计=("seal_amount", "sum"),
    ).reset_index()
    grp = grp[~grp["industry"].isin([None, "未知", ""])]
    if grp.empty:
        return pd.DataFrame()

    out = []
    for _, r in grp.iterrows():
        cnt = int(r["涨停家数"])
        seal = float(r["封板资金合计"] or 0)
        if cnt >= 3:
            st = "🔥强势主线"
        elif cnt >= 2:
            st = "📈升温中"
        elif cnt == 1 and seal >= 2e8:
            st = "🌡️弱势主线"
        else:
            st = "—"
        out.append({
            "板块": str(r["industry"]), "5日涨幅": 0.0, "当日涨幅": 0.0,
            "主力净流入": 0.0, "涨停家数": cnt, "封板资金合计": seal,
            "来源": "涨停池聚合", "状态": st, "强度分": cnt * 10,
        })
    return pd.DataFrame(out).sort_values("涨停家数", ascending=False).reset_index(drop=True)


def get_sector_rotation(zt_df=None):
    rows, seen, chains = [], set(), []

    try:
        d = _fetch_industry_from_em_direct()
        if d is not None and not d.empty:
            for _, r in d.iterrows():
                if r["板块"] not in seen:
                    seen.add(r["板块"]); rows.append(r.to_dict())
            chains.append("push2")
    except Exception:
        pass

    if not rows:
        try:
            d = _fetch_industry_from_sina()
            if d is not None and not d.empty:
                for _, r in d.iterrows():
                    if r["板块"] not in seen:
                        seen.add(r["板块"]); rows.append(r.to_dict())
                chains.append("新浪")
        except Exception:
            pass

    if not rows:
        log("⚠ 外部板块源全挂，启用涨停池聚合兜底", "WARN")
        d = _build_from_limitup(zt_df)
        if not d.empty:
            return d, "兜底"
        return pd.DataFrame(), "失败"

    df = pd.DataFrame(rows).drop_duplicates(subset=["板块"]).copy()
    has_5d = "5日涨幅" in df.columns and df["5日涨幅"].abs().sum() > 0
    has_1d = "当日涨幅" in df.columns and df["当日涨幅"].abs().sum() > 0
    has_net = "主力净流入" in df.columns and df["主力净流入"].abs().sum() > 0
    has_amt = "成交额万" in df.columns and df["成交额万"].abs().sum() > 0
    has_pct = has_5d or has_1d

    if has_1d:
        pct_med = df["当日涨幅"].median()
    elif has_5d:
        pct_med = df["5日涨幅"].median()
    else:
        pct_med = 0.0
    amt_med = df["成交额万"].median() if has_amt else 0.0

    def _get_pct(r):
        if has_5d and r.get("5日涨幅", 0) != 0:
            return r["5日涨幅"], "5日"
        if has_1d:
            return r.get("当日涨幅", 0), "当日"
        return 0, ""

    def strength(r):
        s = 0
        if has_pct:
            pct, kind = _get_pct(r)
            rel = pct - pct_med
            if kind == "当日":
                if pct >= 3: s += 30
                elif pct >= 1.5: s += 20
                elif pct >= 0.5: s += 15
                elif pct >= 0: s += 10
                elif pct >= -1: s += 0
                else: s -= 10
                if rel >= 1.5: s += 10
                elif rel >= 0.5: s += 5
            else:
                if pct >= 5: s += 30
                elif pct >= 2: s += 20
                elif pct >= 0: s += 10
                else: s -= 5
            if has_net:
                net = r["主力净流入"]
                if net > 0: s += 30
                elif net < 0: s -= 15
            elif has_amt:
                amt = r.get("成交额万", 0)
                if amt >= amt_med * 1.5: s += 20
                elif amt >= amt_med: s += 10
                else: s -= 5
        else:
            cnt = r.get("涨停家数", 0)
            s = 50 if cnt >= 5 else 35 if cnt >= 3 else 20 if cnt >= 2 else 10
        return s

    df["强度分"] = df.apply(strength, axis=1)
    strong_thresh = STRONG_THRESHOLD_WITH_NET if has_net else STRONG_THRESHOLD_NO_NET

    def label(r):
        s = r["强度分"]
        if not has_pct:
            cnt = r.get("涨停家数", 0)
            seal = _to_num(r.get("封板资金合计", 0))
            if cnt >= 3: return "🔥强势主线"
            elif cnt >= 2: return "📈升温中"
            elif cnt == 1 and seal >= 2e8: return "🌡️弱势主线"
            return "—"
        net = r["主力净流入"] if has_net else 0
        pct, _ = _get_pct(r)
        if s >= strong_thresh and (not has_net or net > 0):
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

    calib = "5日" if has_5d else ("当日" if has_1d else "兜底")
    src = "+".join(chains) if chains else "无"
    n_s = int((df["状态"] == "🔥强势主线").sum())
    n_w = int((df["状态"] == "📈升温中").sum())
    n_k = int((df["状态"] == "🌡️弱势主线").sum())
    n_d = int((df["状态"] == "📉走弱").sum())
    log(f"✅ 板块主线: {len(df)} 个 | 源={src} | 口径={calib} | 🔥{n_s} 📈{n_w} 🌡️{n_k} 📉{n_d}")

    # v7.2: 记录元信息供推送使用
    df.attrs["has_net"] = has_net
    df.attrs["source"] = src
    df.attrs["calib"] = calib
    return df, "ok"


# ==================== 市场温度 ====================
def calc_market_temperature(zt_df):
    if zt_df is None or zt_df.empty:
        return 0, {"涨停家数": 0, "最高连板": 0, "连板家数": 0, "炸板率": "N/A"}

    zt = len(zt_df)
    cnt_score = 40 if zt >= 80 else 35 if zt >= 60 else 25 if zt >= 40 else 15 if zt >= 20 else 5

    max_b = int(zt_df["board_count"].max()) if "board_count" in zt_df.columns else 1
    multi_b = int((zt_df["board_count"] >= 2).sum()) if "board_count" in zt_df.columns else 0
    ladder = 0
    if max_b >= 4: ladder += 15
    elif max_b >= 3: ladder += 10
    elif max_b >= 2: ladder += 5
    if multi_b >= 10: ladder += 15
    elif multi_b >= 5: ladder += 10
    elif multi_b >= 2: ladder += 5

    if "open_times" in zt_df.columns and zt > 0:
        broken_ratio = int((zt_df["open_times"] > 0).sum()) / zt
    else:
        broken_ratio = 0.3
    br_score = 30 if broken_ratio <= 0.15 else 20 if broken_ratio <= 0.25 else 10 if broken_ratio <= 0.35 else 0

    return cnt_score + ladder + br_score, {
        "涨停家数": zt, "最高连板": max_b, "连板家数": multi_b,
        "炸板率": f"{broken_ratio:.1%}",
    }


def temp_label(t):
    if t >= TEMP_HOT:  return "🔥 热", "满仓操作，主升浪可期"
    if t >= TEMP_WARM: return "☀️ 温", "半仓操作，优先主线龙头"
    if t >= TEMP_COLD: return "🌥️ 冷", "轻仓试探，严格止损"
    return "❄️ 冰冻", "建议空仓观望"


# ==================== 硬过滤 ====================
def hard_filter(df):
    if df.empty:
        return df, {}
    stats = {}

    if HARD_FILTERS["st"] and "name" in df.columns:
        m = df["name"].astype(str).str.contains("ST|退", case=False, na=False)
        if m.any():
            stats["ST"] = int(m.sum()); df = df[~m]

    if "board_count" in df.columns:
        m = df["board_count"] > HARD_FILTERS["max_boards"]
        if m.any():
            stats["连板>3"] = int(m.sum()); df = df[~m]

    if "turnover" in df.columns:
        m = df["turnover"] > HARD_FILTERS["max_turnover"]
        if m.any():
            stats["换手>28%"] = int(m.sum()); df = df[~m]

    if "seal_amount" in df.columns and "circ_market_cap" in df.columns:
        circ = df["circ_market_cap"].replace(0, np.nan)
        ratio = df["seal_amount"] / circ
        m = (df["open_times"].fillna(99) == 0) & (ratio > HARD_FILTERS["seal_ratio_one_word"])
        if m.any():
            stats["一字板"] = int(m.sum()); df = df[~m]

    if "total_market_cap" in df.columns:
        m = df["total_market_cap"] < HARD_FILTERS["min_market_cap"] * 1e8
        if m.any():
            stats[f"市值<{HARD_FILTERS['min_market_cap']}亿"] = int(m.sum()); df = df[~m]

    if "first_time" in df.columns:
        m = df["first_time"].apply(is_late_afternoon)
        if m.any():
            stats["尾盘偷袭"] = int(m.sum()); df = df[~m]

    log(f"🔍 硬过滤: {stats if stats else '无剔除'}")
    return df, stats


# ==================== 评分 ====================
def score_stock(row, sector_counts, use_fund, use_ma):
    global _fund_fast_fail, _kline_fast_fail
    w = WEIGHTS_DEGRADED if (_fund_fast_fail or _kline_fast_fail) else WEIGHTS_NORMAL
    score, max_score = 0, 0
    hits = []

    max_score += 15
    bc = _to_int(row.get("board_count", 1))
    if bc >= 2:
        s = min(bc * w["board_count"], 15)
        score += s
        hits.append(f"{bc}板")

    max_score += w["sector_main"]
    ind = str(row.get("industry", "未知"))
    if sector_counts.get(ind, 0) >= 3:
        s = w["sector_main"]; score += s
        hits.append("板块主线")

    if use_fund and w["fund_flow_5d"] > 0:
        max_score += w["fund_flow_5d"] + w["fund_flow_10d"]
        n5, n10 = get_fund_flow(str(row.get("code", "")).zfill(6))
        if n5 > 0:
            s = min(int(n5 / 1000), w["fund_flow_5d"]); score += s
            # v7.2: 显示具体金额
            hits.append(f"5日资金{fmt_money_short(n5)}")
        if n10 > 0:
            s = min(int(n10 / 2000), w["fund_flow_10d"]); score += s

    max_score += w["morning_lobby"]
    ft = parse_ftime(row.get("first_time"))
    if ft and ft.hour < 10:
        s = w["morning_lobby"]; score += s
        hits.append(f"{ft.strftime('%H:%M')}封板")

    max_score += w["no_open"]
    if _to_int(row.get("open_times", 1)) == 0:
        s = w["no_open"]; score += s
        hits.append("未开板")

    if use_ma and w["ma_bull"] > 0:
        max_score += w["ma_bull"] + w["return_shot"]
        if w["limit_gene"] > 0:
            max_score += w["limit_gene"]
        kl = get_kline(str(row.get("code", "")).zfill(6))
        if check_ma_bull(kl):
            s = w["ma_bull"]; score += s
            hits.append("均线多头")
        if check_return_shot(kl):
            s = w["return_shot"]; score += s
            hits.append("回马枪")
        if w["limit_gene"] > 0 and check_limit_gene(kl, 20):
            s = w["limit_gene"]; score += s
            hits.append("20日涨停基因")

    max_score += w["market_cap_bonus"]
    mcap = _to_num(row.get("total_market_cap", 0))
    if 30 <= mcap / 1e8 <= 100:
        s = w["market_cap_bonus"]; score += s

    max_score += w["turnover_good"]
    to = _to_num(row.get("turnover", 0))
    if 5 <= to <= 15:
        s = w["turnover_good"]; score += s

    max_score += w["seal_amount"]
    if _to_num(row.get("seal_amount", 0)) > 0:
        s = w["seal_amount"]; score += s

    return score, max_score, hits


# ==================== 推送基础 ====================
def send_wecom_webhook(url, content):
    """
    发送到企微。v7.2: 检查长度，超 4000 字节则截断
    """
    if not url:
        log("⚠ WECOM_WEBHOOK 未配置", "WARN")
        return False

    # v7.2: 长度检查
    content_bytes = len(content.encode("utf-8"))
    if content_bytes > WECOM_MAX_BYTES:
        log(f"⚠ 消息 {content_bytes} 字节 > {WECOM_MAX_BYTES} 上限，截断", "WARN")
        # 按字节截断（保留完整行）
        encoded = content.encode("utf-8")
        truncated = encoded[:WECOM_MAX_BYTES].decode("utf-8", errors="ignore")
        # 找到最后一个换行符，保证行完整
        last_nl = truncated.rfind("\n")
        if last_nl > WECOM_MAX_BYTES * 0.5:
            truncated = truncated[:last_nl]
        truncated += "\n\n> ✂️ 内容过长已截断，完整数据见 CSV"
        content = truncated

    try:
        import urllib.request
        req = urllib.request.Request(
            url,
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
    elif net_str:
        net_str = "资金" + net_str

    if cnt > 0 and not pct_str:
        seal = _to_num(r.get("封板资金合计", 0))
        seal_str = f" | 封板{seal/1e8:.2f}亿" if seal > 0 else ""
        return f"- {r['板块']} | 涨停{cnt}家{seal_str}"

    parts = [r['板块']]
    if pct_str: parts.append(pct_str)
    if net_str: parts.append(net_str)
    return f"- {' | '.join(parts)}"


# ==================== 主推送格式化（v7.2 三段式）====================
def format_main_message(df, tag, filtered_stats, sec_df, sec_status,
                        min_score_used, is_degraded, temp, temp_detail):
    now = datetime.now().strftime("%m-%d %H:%M")
    used = df.attrs.get("used_date", "") if hasattr(df, "attrs") else ""

    if used and len(used) == 8:
        try:
            used_str = datetime.strptime(used, "%Y%m%d").strftime("%Y-%m-%d")
        except Exception:
            used_str = used
    else:
        used_str = used or "未知"

    next_str = calc_next_trade_date(used)
    label, advice = temp_label(temp)

    msg = f"## 📈 每日选股 - {now} 生成\n"
    msg += f"**数据日**: {used_str}（收盘）\n"
    msg += f"**建议日**: {next_str}（开盘前参考）\n"

    msg += f"\n### 🌡️ 市场温度：{label} {temp}/100\n"
    msg += f"> 涨停 {temp_detail['涨停家数']} 家 | 最高 {temp_detail['最高连板']} 板 | "
    msg += f"连板 {temp_detail['连板家数']} 家 | 炸板率 {temp_detail['炸板率']}\n"
    msg += f"> **{advice}**\n"

    # ============ 主线监控（v7.2 三段式）============
    if sec_df is not None and not sec_df.empty:
        msg += f"\n### 🌐 主线监控\n"

        # 数据源标注
        is_fallback = sec_status and "兜底" in str(sec_status)
        has_net = sec_df.attrs.get("has_net", False)
        src = sec_df.attrs.get("source", "")
        calib = sec_df.attrs.get("calib", "")

        if is_fallback:
            msg += "> ⚠️ 兜底口径（仅按涨停家数聚合，无涨幅/资金）\n"
        elif not has_net:
            msg += f"> ℹ️ 数据源={src} | 无资金流数据\n"

        # ① 强势主线
        strong = sec_df[sec_df["状态"] == "🔥强势主线"].head(5)
        if not strong.empty:
            msg += "\n**🔥 强势主线（资金抱团）**\n"
            for _, r in strong.iterrows():
                msg += _fmt_sector_line(r) + "\n"
        else:
            msg += "\n**🔥 强势主线**：暂无，市场分散\n"

        # ② 升温中（v7.2 新增）
        warm = sec_df[sec_df["状态"] == "📈升温中"].head(5)
        if not warm.empty:
            msg += "\n**📈 升温中（值得跟踪）**\n"
            for _, r in warm.iterrows():
                msg += _fmt_sector_line(r) + "\n"

        # ③ 走弱预警（v7.2 改为显示 📉走弱，而不是 🌡️弱势主线）
        down = sec_df[sec_df["状态"] == "📉走弱"].head(3)
        if not down.empty:
            msg += "\n**📉 走弱预警（资金撤离）**\n"
            for _, r in down.iterrows():
                msg += _fmt_sector_line(r) + "\n"
    else:
        msg += "\n### 🌐 主线监控\n> ⚠️ 本次未取到板块数据\n"

    # ============ 候选标的 ============
    msg += f"\n### 🏆 候选标的（{len(df)}只）\n"
    if temp < TEMP_COLD:
        msg += "> ❄️ 市场冰冻，建议空仓观望\n"
    elif df.empty:
        msg += "> 今日无符合标准的标的\n"
    else:
        for i, (_, r) in enumerate(df.head(15).iterrows(), 1):
            score = r.get("score", 0)
            name = r.get("name", "")
            code = r.get("code", "")
            ind = r.get("industry", "")
            bc = _to_int(r.get("board_count", 1))
            to = _to_num(r.get("turnover", 0))
            mc = fmt_mcap(r.get("total_market_cap", 0))
            bc_str = f"{bc}板" if bc >= 2 else "首板"
            msg += f"\n**{i}. {name}** `{code}` **{score}分**\n"
            msg += f"> {ind} | {bc_str} | 换手{to:.1f}% | 市值{mc}\n"
            hits = r.get("hits", [])
            if hits:
                msg += f"> 亮点：{' · '.join(hits[:4])}\n"

    # ============ 操作建议 ============
    if temp >= TEMP_HOT:
        no_chase, half_tp, clear_tp = 7.0, 8.0, 12.0
    elif temp >= TEMP_WARM:
        no_chase, half_tp, clear_tp = 5.0, 5.0, 8.0
    else:
        no_chase, half_tp, clear_tp = 3.0, 3.0, 5.0

    msg += f"\n### 📋 操作建议\n"
    msg += f"- **竞价**：高开>{no_chase:.0f}%不追 / 2~{no_chase:.0f}%半仓 / 0~2%正常仓\n"
    msg += f"- **止损**：-3% 无条件\n"
    msg += f"- **止盈**：+{half_tp:.0f}% 减半 / +{clear_tp:.0f}% 清仓\n"
    msg += f"- **时间**：T+2 未表现离场\n"

    if filtered_stats:
        stat_str = " · ".join(f"{k} {v}只" for k, v in filtered_stats.items())
        msg += f"\n> 过滤：{stat_str}\n"
    msg += "\n> 💡 初筛结果，不构成投资建议。"
    return msg


# ==================== 早盘提醒格式化（v7.2 改进）====================
def format_brief_message(df, data_date, days_diff, has_data=True):
    now = datetime.now().strftime("%m-%d %H:%M")
    today_str = datetime.now().strftime("%Y-%m-%d")

    msg = f"## ⏰ 早盘提醒 - {now}\n"
    msg += f"**数据日**: {data_date.strftime('%Y-%m-%d')}（{days_diff}天前）\n"
    msg += f"**今日**: {today_str}\n"

    # v7.2: 数据过期提示
    if days_diff > 10:
        msg += f"> ⚠️ **数据已过期 {days_diff} 天**（跨长假），仅供参考，请以今日盘面为准\n"
        # 长假后不展示候选股，只提示
        msg += f"\n### ⚠️ 长假后特别提示\n"
        msg += f"- 隔夜消息面可能剧变，昨日候选股**不建议直接执行**\n"
        msg += f"- 开盘先观察 15 分钟，看大盘和主线方向\n"
        msg += f"- 有隔夜利好的板块，找低位补涨；无主线，等确定性机会\n"
        msg += f"\n### ⚙️ 通用竞价规则\n"
        msg += f"- 高开 >5%：**不追**\n"
        msg += f"- 高开 2~5%：**半仓**\n"
        msg += f"- 高开 0~2%：**正常仓**\n"
        msg += f"- 低开 >2%：**放弃**\n"
        msg += f"\n> 💡 长假后第一天，仓位保守为上"
        return msg

    if days_diff > 3:
        msg += f"> ℹ️ 数据为 {days_diff} 天前，留意消息面\n"

    # 候选股
    msg += f"\n### 🎯 今日候选\n"
    if not has_data or df.empty:
        msg += "> 昨日无符合标的，建议空仓观望\n"
    else:
        for i, (_, r) in enumerate(df.head(8).iterrows(), 1):
            score = r.get("score", 0)
            name = r.get("name", "")
            code = r.get("code", "")
            ind = r.get("industry", "")
            bc = _to_int(r.get("board_count", 1))
            bc_str = f"{bc}板" if bc >= 2 else "首板"
            msg += f"{i}. **{name}** `{code}` {score}分 —— {ind}·{bc_str}\n"

    msg += f"\n### ⚙️ 竞价规则\n"
    msg += f"- 高开 >5%：**不追**\n"
    msg += f"- 高开 2~5%：**半仓**\n"
    msg += f"- 高开 0~2%：**正常仓**\n"
    msg += f"- 低开 0~-2%：**看 9:30 回封**再决定\n"
    msg += f"- 低开 >2%：**放弃**\n"

    msg += f"\n### ⚠️ 今日检查\n"
    msg += f"- 有无隔夜利空/利好（政策、外盘）\n"
    msg += f"- 目标股竞价量能是否正常\n"
    msg += f"- 大盘竞价氛围（红/绿开）\n"

    msg += f"\n> 💡 本提醒基于历史数据，不构成投资建议"
    return msg


# ==================== 早盘提醒模式（v7.2 改进）====================
def run_brief_mode():
    now = datetime.now()
    today_str = now.strftime("%Y%m%d")

    if not is_trade_date(today_str):
        log(f"📅 {today_str} 非交易日，跳过早盘提醒")
        return

    log(f"⏰ 早盘提醒 | 今日 {today_str} 是交易日")

    out_dir = os.environ.get("OUTPUT_DIR", "results")
    if not os.path.isdir(out_dir):
        log("⚠ 无 results 目录", "WARN")
        return

    csvs = sorted(
        [f for f in os.listdir(out_dir)
         if f.startswith("pick_") and f.endswith("_full.csv")],
        reverse=True
    )
    if not csvs:
        log("⚠ 无历史 full CSV", "WARN")
        return

    latest = csvs[0]
    date_str = latest.replace("pick_", "").replace("_full.csv", "")
    try:
        data_date = datetime.strptime(date_str, "%Y%m%d")
    except Exception:
        log(f"⚠ 无法解析 CSV 日期: {latest}", "WARN")
        return

    days_diff = (now - data_date).days
    if days_diff > BRIEF_MAX_DAYS:
        log(f"⚠ 数据 {days_diff} 天前，超过 {BRIEF_MAX_DAYS} 天，跳过", "WARN")
        return

    log(f"📂 读取 {latest}（{days_diff} 天前）")
    csv_path = os.path.join(out_dir, latest)
    has_data = True
    df = pd.DataFrame()
    try:
        # v7.2: 指定 code 为 str，防前导 0 丢失
        df = pd.read_csv(csv_path, dtype={"code": str})
        if df.empty:
            has_data = False
    except Exception as e:
        log(f"⚠ 读取 CSV 失败: {e}", "WARN")
        has_data = False

    msg = format_brief_message(df, data_date, days_diff, has_data=has_data)
    send_wecom_webhook(os.environ.get("WECOM_WEBHOOK", ""), msg)
    log("✅ 早盘提醒完成")


# ==================== 主推送模式 ====================
def run_full_mode(args):
    global _fund_circuit_broken, _kline_circuit_broken, _fund_fast_fail, _kline_fast_fail

    user_threshold = args.min_score is not None
    min_score = args.min_score if user_threshold else DEFAULT_MIN_SCORE

    # 1. 涨停池
    df_raw = get_limit_up_pool(args.date)
    if df_raw.empty:
        send_wecom_webhook(os.environ.get("WECOM_WEBHOOK", ""),
            "## ⚠️ 选股未执行\n**原因**: 最近交易日无涨停数据")
        return

    # v7.2: 数据日 != 今天时告警
    used_date = df_raw.attrs.get("used_date", "")
    today_str = datetime.now().strftime("%Y%m%d")
    if used_date and used_date != today_str:
        log(f"⚠ 数据日 {used_date} != 今日 {today_str}（可能非交易日或日历延迟）", "WARN")

    # 2. 市场温度
    temp, temp_detail = calc_market_temperature(df_raw)
    label, advice = temp_label(temp)
    log(f"🌡️ 市场温度: {temp}/100 ({label}) | 涨停{temp_detail['涨停家数']}家 "
        f"最高{temp_detail['最高连板']}板 炸板率{temp_detail['炸板率']}")

    # 3. 板块主线
    log("🌐 板块主线监测...")
    sec_df, sec_status = get_sector_rotation(df_raw)
    if not sec_df.empty:
        sec_df.attrs["is_fallback"] = "兜底" in str(sec_status)
        _dump_debug(sec_df, "sector_rotation")

    # 4. 板块统计
    sector_counts = {}
    if "industry" in df_raw.columns:
        sector_counts = df_raw["industry"].value_counts().to_dict()

    # 5. 硬过滤
    df, filtered_stats = hard_filter(df_raw)
    if df.empty:
        send_wecom_webhook(os.environ.get("WECOM_WEBHOOK", ""),
            format_main_message(df, "empty", filtered_stats, sec_df, sec_status,
                                min_score, False, temp, temp_detail))
        return

    # 6. 探针
    log("🔬 探针预检...")
    probe_fund_ok = False
    probe_kline_ok = False
    for _, row in df.head(1).iterrows():
        code = str(row.get("code", "")).zfill(6)
        if not args.no_fund:
            n5, n10 = get_fund_flow(code)
            if n5 != 0 or n10 != 0:
                probe_fund_ok = True
        if not args.no_ma:
            kl = get_kline(code)
            if kl is not None and not kl.empty:
                probe_kline_ok = True

    if not args.no_fund:
        if not probe_fund_ok:
            _fund_fast_fail = True
            log("🚫 资金流完全熔断", "WARN")
        else:
            _fund_circuit_broken = True
            log("🚫 东财资金流熔断（探针成功 → 全走新浪）", "WARN")
    if not args.no_ma:
        if not probe_kline_ok:
            _kline_fast_fail = True
            log("🚫 K线完全熔断", "WARN")
        else:
            _kline_circuit_broken = True
            log("🚫 东财K线熔断（探针成功 → 全走新浪）", "WARN")

    log(f"🔬 预检完成: 资金={('✅' if probe_fund_ok else '❌')} K线={('✅' if probe_kline_ok else '❌')}")

    # 7. 降级
    use_fund = not args.no_fund and not _fund_fast_fail
    use_ma = not args.no_ma and not _kline_fast_fail
    is_degraded = False
    if (_fund_fast_fail or _kline_fast_fail) and not user_threshold:
        min_score = DEGRADED_MIN_SCORE
        is_degraded = True
        log(f"⚠️ 降级模式，阈值→{min_score}")

    # 8. 评分
    log(f"📝 评分 {len(df)} 只（资金={'开' if use_fund else '关'} 均线={'开' if use_ma else '关'} 阈值={min_score}）...")
    t0 = time.time()
    budget_hit = False
    scores, maxes, norms, hits_list = [], [], [], []

    for idx, (_, row) in enumerate(df.iterrows()):
        if time.time() - t0 > SCORE_TIME_BUDGET and not budget_hit:
            budget_hit = True
            if not _fund_fast_fail: _fund_fast_fail = True
            if not _kline_fast_fail: _kline_fast_fail = True
            log(f"⏰ 超预算 {time.time()-t0:.0f}s，剩余走降级", "WARN")
            if not user_threshold:
                min_score = DEGRADED_MIN_SCORE
                is_degraded = True

        _net = not budget_hit and not (_fund_fast_fail and _kline_fast_fail)
        raw, mx, hits = score_stock(
            row, sector_counts,
            use_fund=use_fund and _net and not _fund_fast_fail,
            use_ma=use_ma and _net and not _kline_fast_fail)
        norm = int(round(raw / mx * 100)) if mx > 0 else 0
        scores.append(raw); maxes.append(mx); norms.append(norm); hits_list.append(hits)

    df = df.copy()
    df["raw_score"] = scores
    df["max_score"] = maxes
    df["score"] = norms
    df["hits"] = hits_list
    df_f = df[df["score"] >= min_score].sort_values(
        ["score", "raw_score"], ascending=[False, False]
    ).head(args.top)
    log(f"✅ 达标 {len(df_f)} 只 ≥ {min_score}分 ({time.time()-t0:.1f}s)")

    # 9. 落盘
    out_dir = os.environ.get("OUTPUT_DIR", "results")
    os.makedirs(out_dir, exist_ok=True)
    if not df_f.empty:
        cols = [c for c in ["code", "name", "score", "raw_score", "max_score",
                             "industry", "board_count", "turnover", "total_market_cap"]
                if c in df_f.columns]
        csv_path = f"{out_dir}/pick_{used_date}_full.csv"
        df_f[cols].to_csv(csv_path, index=False, encoding="utf-8-sig")
        log(f"💾 已保存 {len(df_f)}只 → {csv_path}")

    # 10. 推送
    send_wecom_webhook(os.environ.get("WECOM_WEBHOOK", ""),
                      format_main_message(df_f, "full", filtered_stats, sec_df, sec_status,
                                          min_score, is_degraded, temp, temp_detail))
    log("✅ 完成")


# ==================== 主流程 ====================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--brief", action="store_true", help="早盘提醒模式")
    p.add_argument("--no-ma", action="store_true")
    p.add_argument("--no-fund", action="store_true")
    p.add_argument("--top", type=int, default=DEFAULT_TOP_N)
    p.add_argument("--date", type=str, default="")
    p.add_argument("--min-score", type=int, default=None)
    args = p.parse_args()

    is_scheduled = os.environ.get("GITHUB_EVENT_NAME") == "schedule"
    today_str = datetime.now().strftime("%Y%m%d")

    if args.brief:
        log(f"🚀 启动 v7.2 | 模式=brief | 定时={is_scheduled}")
        if is_scheduled and not is_trade_date(today_str):
            log(f"📅 {today_str} 非交易日，跳过早盘提醒")
            return
        try:
            run_brief_mode()
        except Exception as e:
            log(f"❌ 早盘提醒异常: {e}", "ERROR")
            traceback.print_exc()
            sys.exit(1)
        return

    if args.no_ma and args.no_fund:
        mode = "fast"
    elif args.no_fund:
        mode = "no_fund"
    elif args.no_ma:
        mode = "no_ma"
    else:
        mode = "full"

    log(f"🚀 启动 v7.2 | 模式={mode} | 定时={is_scheduled}")

    if is_scheduled and not is_trade_date(today_str):
        log(f"📅 {today_str} 非交易日，跳过主推送")
        return

    try:
        run_full_mode(args)
    except Exception as e:
        log(f"❌ 异常: {e}", "ERROR")
        traceback.print_exc()
        send_wecom_webhook(os.environ.get("WECOM_WEBHOOK", ""),
            f"## ❌ 脚本异常\n**时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n**错误**: {str(e)[:300]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
