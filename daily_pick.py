#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每日选股推送 - 主线增强版 v5.7
修复：探针熔断未正确触发 + 评分循环逐个重试极慢 + 快速失败机制
v5.7 新增：
  1) 全局给 requests 打默认超时补丁，彻底解决 akshare 挂死
  2) 修正 stock_individual_fund_flow 缺 market 参数的调用
  3) retry_with_backoff 增加 ReadTimeout/超时快速失败
  4) 交易日历剔除未来日期
  5) K 线成功时清零失败计数，避免误熔断
"""
# ==================== 全局 requests 超时补丁（必须在 akshare 之前）====================
import requests

_ORIG_SESSION_REQUEST = requests.Session.request


def _patched_session_request(self, method, url, **kwargs):
    if kwargs.get("timeout") is None:
        kwargs["timeout"] = (5, 12)  # (连接超时 5s, 读取超时 12s)
    return _ORIG_SESSION_REQUEST(self, method, url, **kwargs)


requests.Session.request = _patched_session_request
# 兼容 requests.get / requests.post 直调场景
_ORIG_API_REQUEST = requests.api.request


def _patched_api_request(method, url, **kwargs):
    if kwargs.get("timeout") is None:
        kwargs["timeout"] = (5, 12)
    return _ORIG_API_REQUEST(method, url, **kwargs)


requests.api.request = _patched_api_request
requests.get = lambda url, **kw: _patched_api_request("get", url, **kw)
requests.post = lambda url, **kw: _patched_api_request("post", url, **kw)
# ====================================================================================

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

# ==================== 配置 ====================
DEFAULT_TOP_N = 15
RETRY_COUNT = 3
RETRY_DELAY = 5
FUND_FAIL_THRESHOLD = 3
KL_INE_FAIL_THRESHOLD = 5
DEFAULT_MIN_SCORE = 75
DEGRADED_MIN_SCORE = 55

HARD_FILTERS = {
    "st": True, "max_boards": 3, "max_turnover": 28.0,
    "one_word": True, "min_market_cap": 15.0, "late_afternoon": True,
}

WEIGHTS_NORMAL = {
    "board_count": 5, "sector_main": 20, "fund_flow_5d": 15, "fund_flow_10d": 10,
    "morning_lobby": 10, "no_open": 10, "ma_bull": 15, "return_shot": 15,
    "market_cap_bonus": 5, "turnover_good": 5,
}
WEIGHTS_DEGRADED = {
    "board_count": 5, "sector_main": 35, "fund_flow_5d": 0, "fund_flow_10d": 0,
    "morning_lobby": 18, "no_open": 15, "ma_bull": 0, "return_shot": 0,
    "market_cap_bonus": 12, "turnover_good": 15,
}

# ==================== 全局熔断状态 ====================
_fund_circuit_broken = False
_fund_fail_count = 0
_kline_circuit_broken = False
_kline_fail_count = 0
_fund_fast_fail = False
_kline_fast_fail = False


# ==================== 基础工具 ====================
def log(msg: str, level: str = "INFO"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


def retry_with_backoff(func, *args, **kwargs):
    """指数退避重试；连接被断/读超时快速放弃（CI 环境重试意义不大）"""
    global _fund_fast_fail, _kline_fast_fail
    last_error = None
    FAST_FAIL_KEYS = (
        "RemoteDisconnected", "Connection aborted",
        "ReadTimeout", "read timed out", "timed out", "Timeout",
    )
    for i in range(RETRY_COUNT):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_error = e
            err_str = str(e)
            if any(k in err_str for k in FAST_FAIL_KEYS):
                if i >= 1:  # 只重试 1 次就放弃（总共 2 次尝试）
                    log(f"🚫 连接/读超时，快速放弃: {err_str[:60]}", "WARN")
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


def fmt_mcap(yuan_val):
    y = _to_num(yuan_val)
    if y >= 1e8:
        return f"{y/1e8:.1f}亿"
    elif y >= 1e4:
        return f"{y/1e4:.0f}万"
    return f"{y:.0f}元"


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
        # 剔除未来日期（sina 日历偶发脏数据）
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
    for col in ["turnover", "total_market_cap", "circ_market_cap", "board_count", "open_times", "amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col, default in [("board_count", 1), ("turnover", 0.0), ("total_market_cap", 0.0),
                         ("first_time", None), ("open_times", 0), ("industry", "未知")]:
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
        today = datetime.now().strftime("%Y%m%d")
        seen, uniq = set(), []
        for d in [today] + _get_recent_trade_dates(10):
            if d not in seen:
                seen.add(d), uniq.append(d)
        candidates = uniq
        log(f"📡 获取涨停池: 无参回溯，候选 {candidates[:4]}...")

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


# ==================== K线（带熔断+快速失败）====================
def get_kline(code: str, days: int = 60):
    global _kline_circuit_broken, _kline_fail_count, _kline_fast_fail
    if _kline_circuit_broken or _kline_fast_fail:
        return None
    try:
        import akshare as ak
        end = datetime.now()
        start = end - timedelta(days=days * 2)
        code = str(code).zfill(6)
        df = retry_with_backoff(
            ak.stock_zh_a_hist,
            symbol=code, period="daily",
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
            adjust="qfq",
        )
        if df is not None and not df.empty:
            if "日期" in df.columns:
                df["日期"] = pd.to_datetime(df["日期"])
                df = df.sort_values("日期")
            _kline_fail_count = 0  # 成功清零，避免误熔断
            return df.tail(days)
    except Exception as e:
        log(f"⚠ {code} K线失败: {str(e)[:60]}", "WARN")

    _kline_fail_count += 1
    if _kline_fail_count >= KL_INE_FAIL_THRESHOLD:
        _kline_circuit_broken = True
        log(f"🚫 K线连续失败{_kline_fail_count}只，已熔断", "WARN")
    return None


def check_ma_bull(df):
    if df is None or len(df) < 22 or "收盘" not in df.columns:
        return False
    try:
        c = df["收盘"]
        ma5, ma10, ma20 = c.rolling(5).mean().iloc[-1], c.rolling(10).mean().iloc[-1], c.rolling(20).mean().iloc[-1]
        return ma5 > ma10 > ma20
    except Exception:
        return False


def check_return_shot(df):
    if df is None or len(df) < 10 or "收盘" not in df.columns:
        return False
    try:
        c = df["收盘"]
        ret = c.pct_change() * 100
        if not (ret >= 9.8).any():
            return False
        h, l = c.tail(10).max(), c.tail(10).min()
        return (h - l) / h >= 0.05 if h > 0 else False
    except Exception:
        return False


# ==================== 资金流向（带熔断+快速失败）====================
def get_fund_flow(code: str):
    global _fund_circuit_broken, _fund_fail_count, _fund_fast_fail
    if _fund_circuit_broken or _fund_fast_fail:
        return 0.0, 0.0
    code = str(code).zfill(6)
    try:
        import akshare as ak
        mkt = "sh" if code.startswith(("6", "9")) else "sz"
        df = retry_with_backoff(ak.stock_individual_fund_flow, stock=code, market=mkt)
        if df is None or df.empty:
            _fund_fail_count += 1
            return 0.0, 0.0
        col = _find_col(df, "主力净流入")
        if not col:
            _fund_fail_count += 1
            return 0.0, 0.0
        df = df.sort_values(df.columns[0], ascending=False)
        net_5 = pd.to_numeric(df[col].head(5), errors="coerce").sum()
        net_10 = pd.to_numeric(df[col].head(10), errors="coerce").sum()
        _fund_fail_count = 0  # 成功清零
        return float(net_5), float(net_10)
    except Exception as e:
        _fund_fail_count += 1
        log(f"⚠ 资金流失败 {code}: {str(e)[:60]}", "WARN")
        if _fund_fail_count >= FUND_FAIL_THRESHOLD:
            _fund_circuit_broken = True
            log(f"🚫 资金流向连续失败{_fund_fail_count}只，已熔断", "WARN")
        return 0.0, 0.0


# ==================== 板块主线监测 ====================
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
        out.append({"板块": str(r[name_col]), "5日涨幅": _to_num(r.get(pct_col, 0)) if pct_col else 0.0,
                    "主力净流入": _to_num(r.get(net_col, 0)) if net_col else 0.0, "来源": "同花顺"})
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
        out.append({"板块": str(r[name_col]), "5日涨幅": _to_num(r.get(pct_col, 0)) if pct_col else 0.0,
                    "主力净流入": _to_num(r.get(net_col, 0)) if net_col else 0.0, "来源": "东财"})
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
        out.append({"板块": str(r[name_col]), "5日涨幅": _to_num(r.get(pct_col, 0)) if pct_col else 0.0,
                    "主力净流入": _to_num(r.get(net_col, 0)) if net_col else 0.0, "来源": "东财资金流"})
    log(f"✅ 东财资金流排名: {len(out)} 个板块")
    return pd.DataFrame(out)


def _build_from_limitup(zt_df):
    if zt_df is None or zt_df.empty or "industry" not in zt_df.columns:
        return pd.DataFrame()
    vc = zt_df["industry"].value_counts()
    out = []
    for name, cnt in vc.items():
        if name in (None, "未知", ""):
            continue
        if cnt >= 3:
            status = "🔥强势主线"
        elif cnt >= 2:
            status = "📈升温中"
        else:
            status = "—"
        out.append({"板块": str(name), "5日涨幅": 0.0, "主力净流入": 0.0,
                    "涨停家数": int(cnt), "来源": "涨停池聚合", "状态": status, "强度分": cnt * 10})
    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out).sort_values("涨停家数", ascending=False).reset_index(drop=True)
    log(f"✅ 涨停池聚合主线: {len(df)} 个板块，TOP3: {' | '.join(f'{r.板块}({r.涨停家数}家)' for r in df.head(3).itertuples())}")
    return df


def get_sector_rotation(zt_df=None):
    rows = []
    status = "ok"
    seen = set()

    try:
        d = _fetch_industry_from_ths()
        if d is not None and not d.empty:
            for _, r in d.iterrows():
                if r["板块"] not in seen:
                    seen.add(r["板块"]); rows.append(r.to_dict())
    except Exception as e:
        log(f"⚠ 同花顺源失败: {e}", "WARN"); status = "同花顺源失败"

    try:
        d = _fetch_industry_from_em()
        if d is not None and not d.empty:
            for _, r in d.iterrows():
                if r["板块"] not in seen:
                    seen.add(r["板块"]); rows.append(r.to_dict())
    except Exception as e:
        log(f"⚠ 东财行业榜失败: {e}", "WARN")
        if status == "ok": status = "东财行业榜失败"

    try:
        d = _fetch_fund_flow_rank()
        if d is not None and not d.empty:
            for _, r in d.iterrows():
                if r["板块"] in seen:
                    for existing in rows:
                        if existing["板块"] == r["板块"] and r["主力净流入"] != 0:
                            existing["主力净流入"] = r["主力净流入"]
                            if r["5日涨幅"] != 0:
                                existing["5日涨幅"] = r["5日涨幅"]
                            break
                else:
                    seen.add(r["板块"]); rows.append(r.to_dict())
    except Exception as e:
        log(f"⚠ 东财资金流排名失败: {e}", "WARN")
        if status == "ok": status = "资金流排名不可用"

    if not rows:
        log("⚠ 外部板块源全挂，启用涨停池聚合兜底", "WARN")
        d = _build_from_limitup(zt_df)
        if not d.empty:
            return d, "涨停池聚合兜底（无外部资金流）"
        return pd.DataFrame(), "全部失败"

    df = pd.DataFrame(rows).drop_duplicates(subset=["板块"]).copy()
    has_real_pct = df["5日涨幅"].abs().sum() > 0

    def calc_strength(r):
        if has_real_pct:
            score = 0
            if r["5日涨幅"] >= 5:      score += 30
            elif r["5日涨幅"] >= 2:    score += 20
            elif r["5日涨幅"] >= 0:    score += 10
            if r["主力净流入"] > 0:    score += 30
            elif r["主力净流入"] < 0:  score -= 15
            return score
        else:
            cnt = r.get("涨停家数", 0)
            if cnt >= 5: return 50
            elif cnt >= 3: return 35
            elif cnt >= 2: return 20
            return 10

    df["强度分"] = df.apply(calc_strength, axis=1)

    def label(r):
        if not has_real_pct:
            cnt = r.get("涨停家数", 0)
            if cnt >= 3: return "🔥强势主线"
            elif cnt >= 2: return "📈升温中"
            return "—"
        if r["强度分"] >= 60 and r["主力净流入"] > 0: return "🔥强势主线"
        if r["强度分"] >= 40: return "📈升温中"
        if r["强度分"] < 0 and r["主力净流入"] < 0: return "📉走弱"
        if r["5日涨幅"] > 3 and r["主力净流入"] < 0: return "⚠️脉冲"
        return "—"

    df["状态"] = df.apply(label, axis=1)
    df = df.sort_values(["强度分", "主力净流入"], ascending=False).reset_index(drop=True)
    log(f"✅ 板块主线分析完成: {len(df)} 个板块 | status={status}")
    return df, status


# ==================== 过滤 & 评分 ====================
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
        ratio = df["seal_amount"] / (circ * 1e8)
        mask = (df["open_times"].fillna(99) == 0) & (ratio > 0.08)
        if mask.any():
            reasons.append(f"一字板: {mask.sum()}只"); df = df[~mask]
    if "total_market_cap" in df.columns:
        mask = df["total_market_cap"] < HARD_FILTERS["min_market_cap"]
        if mask.any():
            reasons.append(f"市值<15亿: {mask.sum()}只"); df = df[~mask]
    if HARD_FILTERS["late_afternoon"] and "first_time" in df.columns:
        mask = df["first_time"].apply(lambda x: is_late_afternoon(x))
        if mask.any():
            reasons.append(f"尾盘偷袭: {mask.sum()}只"); df = df[~mask]
    log(f"🔍 硬过滤: 剔除{original - len(df)}只 ({', '.join(reasons)})")
    return df, reasons


def score_stock(row, sector_counts, use_fund, use_ma):
    global _fund_circuit_broken, _kline_circuit_broken
    weights = WEIGHTS_DEGRADED if (_fund_circuit_broken or _kline_circuit_broken) else WEIGHTS_NORMAL
    score = 0
    reasons = []

    bc = _to_int(row.get("board_count", 1))
    if bc >= 2:
        s = min(bc * weights["board_count"], 15)
        score += s; reasons.append(f"连板{bc}层(+{s})")

    ind = str(row.get("industry", "未知"))
    if sector_counts.get(ind, 0) >= 3:
        s = weights["sector_main"]
        score += s; reasons.append(f"主线({ind}{sector_counts[ind]}家)(+{s})")

    if use_fund and weights["fund_flow_5d"] > 0:
        n5, n10 = get_fund_flow(str(row.get("code", "")).zfill(6))
        if n5 > 0:
            s = min(int(n5 / 1000), weights["fund_flow_5d"])
            score += s; reasons.append(f"5日+{n5:.0f}万(+{s})")
        if n10 > 0:
            s = min(int(n10 / 2000), weights["fund_flow_10d"])
            score += s; reasons.append(f"10日+{n10:.0f}万(+{s})")

    ft = parse_ftime(row.get("first_time"))
    if ft and ft.hour < 10:
        s = weights["morning_lobby"]
        score += s; reasons.append(f"早盘{ft.strftime('%H:%M')}(+{s})")

    if _to_int(row.get("open_times", 1)) == 0:
        s = weights["no_open"]; score += s; reasons.append(f"未开板(+{s})")

    if use_ma and weights["ma_bull"] > 0 and not _kline_circuit_broken:
        kl = get_kline(str(row.get("code", "")).zfill(6))
        if check_ma_bull(kl):
            s = weights["ma_bull"]; score += s; reasons.append(f"均线多头(+{s})")
        if check_return_shot(kl):
            s = weights["return_shot"]; score += s; reasons.append(f"回马枪(+{s})")

    mcap = _to_num(row.get("total_market_cap", 0))
    if 30 <= mcap / 1e8 <= 100:
        s = weights["market_cap_bonus"]; score += s; reasons.append(f"市值{fmt_mcap(mcap)}(+{s})")

    to = _to_num(row.get("turnover", 0))
    if 5 <= to <= 15:
        s = weights["turnover_good"]; score += s; reasons.append(f"换手{to:.1f}%健康(+{s})")

    return min(score, 100), reasons


# ==================== 企微推送 ====================
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


def format_sector_section(sec_df, status):
    if sec_df is None or sec_df.empty:
        return "\n### 🌐 板块主线监测\n> ⚠️ 本次未取到任何板块数据，主线判断暂缺\n"
    msg = "\n### 🌐 板块主线监测\n"
    if status and status != "ok":
        msg += f"> 状态: {status}\n"
    strong = sec_df[sec_df["状态"] == "🔥强势主线"].head(6)
    if not strong.empty:
        msg += "\n**🔥 强势主线 / 涨停抱团**\n"
        for _, r in strong.iterrows():
            if r.get("涨停家数", 0) > 0 and r.get("主力净流入", 0) == 0:
                msg += f"- {r['板块']} | 涨停{r['涨停家数']}家（兜底口径）\n"
            else:
                net = r['主力净流入']
                net_str = f"{net/1e8:+.2f}亿" if abs(net) >= 1e8 else f"{net/1e4:+.0f}万"
                msg += f"- {r['板块']} | 5日{r['5日涨幅']:+.1f}% | 资金{net_str}\n"
    else:
        msg += "> 暂无达到阈值的强势主线\n"
    return msg


def format_message(df, mode, tag, filtered_reasons, sec_df, sec_status, min_score_used):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    used = df.attrs.get("used_date", "") if hasattr(df, "attrs") else ""
    used_line = f" | 数据日:{used}" if used else ""

    msg = f"## 📈 每日选股推送 - {tag}\n**时间**: {now}{used_line} | **模式**: {mode}\n"
    msg += f"**阈值**: {min_score_used}分（{'降级模式' if min_score_used != DEFAULT_MIN_SCORE else '正常'})\n"
    msg += format_sector_section(sec_df, sec_status)
    msg += f"\n### 🏆 候选标的（≥{min_score_used}分，{len(df)}只）\n"

    if df.empty:
        msg += "> 今日无符合标准的标的，建议空仓观望。\n"
    else:
        for i, (_, r) in enumerate(df.head(15).iterrows(), 1):
            msg += f"\n**{i}. {r.get('name','')}** (`{r.get('code','')}`)\n"
            msg += f"- 评分:**{r.get('score',0)}** | 板块:{r.get('industry','N/A')} | 连板:{_to_int(r.get('board_count',1))}\n"
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
    global _fund_circuit_broken, _kline_circuit_broken, _fund_fail_count, _kline_fail_count

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
    log(f"🚀 启动 v5.7 | 模式={mode} | 日期={args.date or '今日(回溯)'}")

    try:
        min_score = args.min_score if args.min_score is not None else DEFAULT_MIN_SCORE

        # 1. 涨停池
        df_raw = get_limit_up_pool(args.date)
        if df_raw.empty:
            send_wecom_webhook(os.environ.get("WECOM_WEBHOOK", ""),
                "## ⚠️ 选股未执行\n**时间**: " + datetime.now().strftime("%Y-%m-%d %H:%M") +
                "\n**原因**: 最近交易日涨停池均为空")
            sys.exit(1)

        # 2. 板块主线
        log("🌐 板块主线监测...")
        sec_df, sec_status = get_sector_rotation(df_raw)
        log(f"🌐 板块数据 rows={len(sec_df)} | status={sec_status}")
        if not sec_df.empty:
            top3_parts = []
            for _, r in sec_df.head(3).iterrows():
                st = r.get("状态", "—")
                top3_parts.append(f"{r['板块']}({st})")
            log(f"✅ 板块TOP3: {' | '.join(top3_parts)}")
            _dump_debug(sec_df, "sector_rotation_result")

        # 3. 板块统计
        sector_counts = {}
        if "industry" in df_raw.columns:
            sector_counts = df_raw["industry"].value_counts().to_dict()
            top3 = sorted(sector_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            log(f"📊 涨停板块TOP3: {' | '.join(f'{k}({v})' for k,v in top3)}")

        # 4. 硬过滤
        df, filtered_reasons = hard_filter(df_raw)
        if df.empty:
            log("📭 过滤后无标的")
            send_wecom_webhook(os.environ.get("WECOM_WEBHOOK", ""),
                              format_message(df, mode, tag, filtered_reasons, sec_df, sec_status, min_score))
            return

        # 5. 探针预检
        log("🔬 探针预检接口可用性...")
        probe = df.head(min(3, len(df)))
        for _, row in probe.iterrows():
            code = str(row.get("code", "")).zfill(6)
            get_fund_flow(code)
            get_kline(code)

        # 5.5 强制检查熔断（探针 3 只全失败时确保熔断标志被设置）
        if _fund_fail_count >= FUND_FAIL_THRESHOLD and not _fund_circuit_broken:
            _fund_circuit_broken = True
            log(f"🚫 探针后强制熔断资金接口（失败{_fund_fail_count}次）", "WARN")
        if _kline_fail_count >= KL_INE_FAIL_THRESHOLD and not _kline_circuit_broken:
            _kline_circuit_broken = True
            log(f"🚫 探针后强制熔断K线接口（失败{_kline_fail_count}次）", "WARN")

        log(f"🔬 预检完成: 资金熔断={_fund_circuit_broken} K线熔断={_kline_circuit_broken}")

        # 6. 降级判定
        use_fund = not args.no_fund and not _fund_circuit_broken
        use_ma = not args.no_ma and not _kline_circuit_broken
        if (_fund_circuit_broken or _kline_circuit_broken) and args.min_score is None:
            min_score = DEGRADED_MIN_SCORE
            log(f"⚠️ 接口降级，阈值自动从{DEFAULT_MIN_SCORE}调整为{min_score}", "WARN")

        # 7. 评分
        log(f"📝 开始评分 {len(df)} 只（资金={'开' if use_fund else '关'} 均线={'开' if use_ma else '关'} 阈值={min_score}）...")
        t0 = time.time()
        scores, all_reasons = [], []
        for idx, (_, row) in enumerate(df.iterrows()):
            s, rs = score_stock(row, sector_counts, use_fund=use_fund, use_ma=use_ma)
            scores.append(s); all_reasons.append(rs)
            if (idx + 1) % 10 == 0:
                elapsed = time.time() - t0
                log(f"  ⏳ {idx+1}/{len(df)} 只 ({elapsed:.1f}s)")

        df = df.copy()
        df["score"] = scores
        df["reasons"] = all_reasons
        df_f = df[df["score"] >= min_score].sort_values("score", ascending=False).head(args.top)
        log(f"✅ 达标 {len(df_f)} 只 ≥ {min_score}分 ({time.time()-t0:.1f}s)")

        # 8. 落盘
        out_dir = os.environ.get("OUTPUT_DIR", "results")
        os.makedirs(out_dir, exist_ok=True)
        if not df_f.empty:
            cols = [c for c in ["code", "name", "score", "industry", "board_count", "turnover", "total_market_cap"] if c in df_f.columns]
            df_f[cols].to_csv(f"{out_dir}/pick_{datetime.now().strftime('%Y%m%d')}_{mode}.csv",
                              index=False, encoding="utf-8-sig")
            log(f"💾 已保存 {len(df_f)}只")

        # 9. 推送
        send_wecom_webhook(os.environ.get("WECOM_WEBHOOK", ""),
                          format_message(df_f, mode, tag, filtered_reasons, sec_df, sec_status, min_score))
        log("✅ 选股完成")

    except Exception as e:
        log(f"❌ 异常: {e}", "ERROR")
        traceback.print_exc()
        send_wecom_webhook(os.environ.get("WECOM_WEBHOOK", ""),
            f"## ❌ 脚本异常\n**时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n**错误**: {str(e)[:300]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
