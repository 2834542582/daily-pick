#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每日选股推送 - 主线增强版 v5.8
===========================================================
v5.8 改进（相对 v5.7）：
  1) 超时参数收紧：(5,12) → (3,6)，重试等待 5s → 2s
  2) 超时直接快速放弃（i>=0），不再重试一次
  3) 探针预检从 3 只 → 1 只，减少探针阶段耗时
  4) _fund_fast_fail / _kline_fast_fail 在熔断时同步置位，
     后续调用直接入口返回，连一次请求都不发
  5) 新增「回马枪」精细判定（昨涨停 + 今缩量 + 不破 MA5 + 今日小实体）
  6) 新增「涨停基因」加分（近 20 日曾涨停）
  7) 新增「封板资金」加分
  8) 全部评分项补充数据口径说明

===========================================================
数据口径速查（每次改参数先看这里）：
-----------------------------------------------------------
【涨停池 stock_zt_pool_em】
  - 数据源：东方财富涨停板行情
  - 时间口径：T 日收盘后，含当日最终封板状态
  - 连板数：含当日，首板=1，2 板=2……
  - 换手率：当日实际换手率（%）
  - 流通市值 / 总市值：单位「元」，脚本里除 1e8 转「亿」
  - 封板资金：单位为「元」，即当日封单金额
  - 首次封板时间：HHMMSS 六位数字，如 093015 = 09:30:15
  - 炸板次数：当日盘中开板次数，0 = 未开板（一字/T字）
  - 所属行业：东财行业分类（非申万）

【资金流 stock_individual_fund_flow】
  - 数据源：东方财富个股资金流向
  - 「主力净流入」= 超大单净额 + 大单净额（不含中单/小单）
  - 本脚本用「近 5 日累计 / 近 10 日累计」，非单日
  - 单位：元；脚本里 /1e4 显示为「万」

【K 线 stock_zh_a_hist】
  - 数据源：东方财富日线（前复权 adjust="qfq"）
  - MA5/MA10/MA20：收盘价简单移动平均，前复权口径
  - 涨停近似：单日涨幅 ≥ 9.8%（因四舍五入，主板 10% 涨停约等于 9.95%~10.05%）
  - 回马枪判定窗口：近 10 日

【板块主线 get_sector_rotation】
  - 优先序：同花顺行业榜 → 东财行业榜 → 东财资金流排名
  - 全部失败时兜底：按涨停池「所属行业」聚合，涨停≥3家记强势主线
  - 兜底口径下「主力净流入」为 0（无法获取），只统计涨停家数

【评分权重】
  - 正常模式 WEIGHTS_NORMAL：满分 100
  - 降级模式 WEIGHTS_DEGRADED：当资金流或 K 线熔断时启用，
    把依赖接口的权重转移到板块/连板/换手等本地可算项
  - 阈值：正常 75 分，降级 55 分
===========================================================
"""
# ==================== 全局 requests 超时补丁（必须在 akshare 之前）====================
import requests

_ORIG_SESSION_REQUEST = requests.Session.request
_TIMEOUT = (3, 6)  # (连接超时 3s, 读取超时 6s)


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
RETRY_COUNT = 2                 # v5.8: 3 → 2
RETRY_DELAY = 2                 # v5.8: 5 → 2
FUND_FAIL_THRESHOLD = 2         # v5.8: 3 → 2
KL_INE_FAIL_THRESHOLD = 3       # v5.8: 5 → 3
DEFAULT_MIN_SCORE = 75
DEGRADED_MIN_SCORE = 55
PROBE_N = 1                     # v5.8: 探针只跑 1 只

# 硬过滤阈值（口径见文件头 docstring）
HARD_FILTERS = {
    "st": True,                 # 剔除名称含 ST / 退
    "max_boards": 3,            # 连板数 > 3 剔除（高位接力风险）
    "max_turnover": 28.0,       # 换手率 > 28% 剔除（过度换手）
    "one_word": True,           # 一字板剔除（封单/流通市值 > 8% 且未开板）
    "min_market_cap": 15.0,     # 总市值 < 15 亿剔除（单位：亿元）
    "late_afternoon": True,     # 14:30 后首次封板剔除（尾盘偷袭）
}

# 评分权重（正常模式，满分 100）
WEIGHTS_NORMAL = {
    "board_count": 5,           # 连板数：每层 5 分，上限 15 分
    "sector_main": 20,          # 主线加分：所属行业涨停 ≥ 3 家
    "fund_flow_5d": 15,         # 5 日主力净流入：每 1000 万 1 分，上限 15 分
    "fund_flow_10d": 10,        # 10 日主力净流入：每 2000 万 1 分，上限 10 分
    "morning_lobby": 10,        # 早盘封板：首次封板 < 10:00
    "no_open": 10,              # 未开板：炸板次数 = 0
    "ma_bull": 15,              # 均线多头：MA5>MA10>MA20
    "return_shot": 15,          # 回马枪：昨涨停+今缩量+不破MA5+今小实体
    "market_cap_bonus": 5,      # 市值 30~100 亿（中盘股，弹性与稳健平衡）
    "turnover_good": 5,         # 换手 5%~15%（活跃但不失控）
    "seal_amount": 5,           # v5.8 新增：封板资金 > 0（有封单托底）
    "limit_gene": 10,           # v5.8 新增：近 20 日曾涨停（涨停基因）
}

# 降级模式权重（资金流或 K 线熔断时启用）
WEIGHTS_DEGRADED = {
    "board_count": 5,           # 保留（本地可算）
    "sector_main": 30,          # 20 → 30（补足）
    "fund_flow_5d": 0,          # 熔断，置零
    "fund_flow_10d": 0,         # 熔断，置零
    "morning_lobby": 20,        # 10 → 20（补足）
    "no_open": 15,              # 10 → 15（补足）
    "ma_bull": 0,               # 熔断，置零
    "return_shot": 0,           # 熔断，置零
    "market_cap_bonus": 10,     # 5 → 10（补足）
    "turnover_good": 10,        # 5 → 10（补足）
    "seal_amount": 10,          # v5.8: 5 → 10（本地可算）
    "limit_gene": 0,            # 熔断（依赖 K 线），置零
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
    """
    指数退避重试。
    v5.8: 连接被断 / 读超时 → 第一次就放弃（i>=0），不再重试。
    原因：CI 环境对同一域名的超时通常是持续性故障，重试纯浪费时间。
    """
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
                if i >= 0:  # v5.8: 从 i>=1 改为 i>=0，第一次超时就放弃
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
    """字符串 → float，失败返回 0.0。用于兼容 '1,234' / '12.3%' 等格式。"""
    try:
        return float(str(x).replace(",", "").replace("%", "").strip())
    except Exception:
        return 0.0


def _to_int(x):
    """字符串 → int，失败返回 1（连板数默认 1）。"""
    try:
        return int(float(str(x).replace(",", "").strip()))
    except Exception:
        return 1


def fmt_mcap(yuan_val):
    """市值格式化：元 → 亿 / 万 / 元"""
    y = _to_num(yuan_val)
    if y >= 1e8:
        return f"{y/1e8:.1f}亿"
    elif y >= 1e4:
        return f"{y/1e4:.0f}万"
    return f"{y:.0f}元"


def _find_col(df, *keywords):
    """模糊匹配列名，返回第一个命中列。用于兼容不同数据源列名差异。"""
    for c in df.columns:
        for kw in keywords:
            if kw in str(c):
                return c
    return None


def parse_ftime(ftime_str):
    """
    解析「首次封板时间」。
    口径：东财返回 HHMMSS 六位数字（如 093015 = 09:30:15）
    兼容：也接受 'HH:MM:SS' 字符串
    返回：datetime 对象（日期部分固定为 2020-01-01，只用于比较时刻）
    """
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
    """尾盘偷袭判定：首次封板时间 > 14:30"""
    dt = parse_ftime(ftime_str)
    if dt is None:
        return False
    return dt > datetime(dt.year, dt.month, dt.day, 14, 30)


# ==================== 涨停池 ====================
def _get_recent_trade_dates(n: int = 10) -> List[str]:
    """
    取最近 n 个交易日（含今日）。
    口径：新浪交易日历 tool_trade_date_hist_sina，剔除未来日期。
    """
    try:
        import akshare as ak
        cal = retry_with_backoff(ak.tool_trade_date_hist_sina)
        if cal is None or cal.empty or "trade_date" not in cal.columns:
            return []
        cal = cal.copy()
        cal["trade_date"] = pd.to_datetime(cal["trade_date"])
        cal = cal[cal["trade_date"] <= pd.Timestamp.today()]   # 剔除未来脏数据
        cal = cal.sort_values("trade_date", ascending=False)
        return [d.strftime("%Y%m%d") for d in cal["trade_date"].head(n)]
    except Exception as e:
        log(f"⚠ 交易日历失败: {e}", "WARN")
        return []


def _normalize(df: pd.DataFrame, used_date: str) -> pd.DataFrame:
    """
    列名标准化。
    口径对照（东财原始 → 本脚本内部名）：
      代码        → code
      名称        → name
      涨跌幅      → pct_change
      最新价      → price
      成交额      → amount
      流通市值    → circ_market_cap（单位：元）
      总市值      → total_market_cap（单位：元）
      换手率      → turnover（%）
      封板资金    → seal_amount（单位：元）
      首次封板时间 → first_time（HHMMSS）
      最后封板时间 → last_time
      炸板次数    → open_times
      连板数      → board_count（含当日）
      所属行业    → industry（东财口径，非申万）
    """
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
    for col in ["turnover", "total_market_cap", "circ_market_cap", "board_count", "open_times", "amount", "seal_amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col, default in [("board_count", 1), ("turnover", 0.0), ("total_market_cap", 0.0),
                         ("first_time", None), ("open_times", 0), ("industry", "未知"),
                         ("seal_amount", 0.0)]:
        if col not in df.columns:
            df[col] = default
    df.attrs["used_date"] = used_date
    return df


def get_limit_up_pool(trade_date: str = None) -> pd.DataFrame:
    """
    取涨停池。
    口径：东财 stock_zt_pool_em，T 日收盘后数据（含当日最终封板状态）。
    无指定日期时回溯最近 10 个交易日，取第一个非空的。
    """
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
    """
    取 K 线（前复权日线，60 日）。
    口径：东财 stock_zh_a_hist，adjust="qfq"（前复权）。
    熔断：连续失败 KL_INE_FAIL_THRESHOLD 只 → _kline_circuit_broken=True
    """
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
            _kline_fail_count = 0   # 成功清零，避免误熔断
            return df.tail(days)
    except Exception as e:
        log(f"⚠ {code} K线失败: {str(e)[:60]}", "WARN")

    _kline_fail_count += 1
    if _kline_fail_count >= KL_INE_FAIL_THRESHOLD:
        _kline_circuit_broken = True
        _kline_fast_fail = True     # v5.8: 同步置位，后续调用入口直接返回
        log(f"🚫 K线连续失败{_kline_fail_count}只，已熔断", "WARN")
    return None


def check_ma_bull(df) -> bool:
    """
    均线多头口径：
      - MA5 > MA10 > MA20（收盘价简单移动平均，前复权）
      - 数据不足 22 根时返回 False
    """
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
    """
    回马枪口径（v5.8 精细化）：
      1) 昨日涨幅 ≥ 9.8%（近似涨停，主板 10%）
      2) 今日成交量 < 昨日成交量（缩量）
      3) 今日收盘 > MA5（不破 5 日线）
      4) 今日振幅 < 3%（小实体，非再接大阳/大阴）
    """
    if df is None or len(df) < 10 or "收盘" not in df.columns or "成交量" not in df.columns:
        return False
    try:
        c = df["收盘"]
        v = df["成交量"]
        # 昨涨停
        up_yest = (c.iloc[-2] / c.iloc[-3] - 1) >= 0.095
        # 今缩量
        shrink = v.iloc[-1] < v.iloc[-2]
        # 不破 MA5
        ma5 = c.rolling(5).mean().iloc[-1]
        hold_ma = c.iloc[-1] > ma5
        # 今日小实体
        today_chg = abs(c.iloc[-1] / c.iloc[-2] - 1)
        is_small = today_chg < 0.03
        return up_yest and shrink and hold_ma and is_small
    except Exception:
        return False


def check_limit_gene(df, days: int = 20) -> bool:
    """
    涨停基因口径（v5.8 新增）：
      近 20 日（不含当日）曾出现单日涨幅 ≥ 9.8%
    """
    if df is None or len(df) < days + 1 or "收盘" not in df.columns:
        return False
    try:
        recent = df.tail(days + 1).iloc[:-1]   # 剔除当日
        ret = recent["收盘"].pct_change() * 100
        return (ret >= 9.8).any()
    except Exception:
        return False


# ==================== 资金流向（带熔断+快速失败）====================
def get_fund_flow(code: str):
    """
    取个股资金流。
    口径：东财 stock_individual_fund_flow
      - 「主力净流入」= 超大单净额 + 大单净额（不含中单/小单）
      - 返回 (近5日累计, 近10日累计)，单位：元
    熔断：连续失败 FUND_FAIL_THRESHOLD 只 → _fund_circuit_broken=True
    """
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
        df = df.sort_values(df.columns[0], ascending=False)   # 第一列是日期，倒序
        net_5 = pd.to_numeric(df[col].head(5), errors="coerce").sum()
        net_10 = pd.to_numeric(df[col].head(10), errors="coerce").sum()
        _fund_fail_count = 0   # 成功清零
        return float(net_5), float(net_10)
    except Exception as e:
        _fund_fail_count += 1
        log(f"⚠ 资金流失败 {code}: {str(e)[:60]}", "WARN")
        if _fund_fail_count >= FUND_FAIL_THRESHOLD:
            _fund_circuit_broken = True
            _fund_fast_fail = True   # v5.8: 同步置位
            log(f"🚫 资金流向连续失败{_fund_fail_count}只，已熔断", "WARN")
        return 0.0, 0.0


# ==================== 板块主线监测 ====================
def _fetch_industry_from_ths():
    """同花顺行业榜（口径：板块名称 + 涨跌幅 + 主力净流入）"""
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
                    "主力净流入": _to_num(r.get(net_col, 0)) if net_col else 0.0,
                    "来源": "同花顺"})
    log(f"✅ 同花顺行业源: {len(out)} 个板块")
    return pd.DataFrame(out)


def _fetch_industry_from_em():
    """东财行业榜（口径：板块名称 + 涨跌幅 + 主力净流入）"""
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
                    "主力净流入": _to_num(r.get(net_col, 0)) if net_col else 0.0,
                    "来源": "东财"})
    log(f"✅ 东财行业源: {len(out)} 个板块")
    return pd.DataFrame(out)


def _fetch_fund_flow_rank():
    """东财板块资金流排名（5日口径）"""
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
                    "主力净流入": _to_num(r.get(net_col, 0)) if net_col else 0.0,
                    "来源": "东财资金流"})
    log(f"✅ 东财资金流排名: {len(out)} 个板块")
    return pd.DataFrame(out)


def _build_from_limitup(zt_df):
    """
    兜底口径（v5.8）：
      当外部板块源全挂时，按涨停池「所属行业」聚合。
      涨停 ≥ 3 家 → 🔥强势主线
      涨停 = 2 家 → 📈升温中
      涨停 = 1 家 → —
      此时「主力净流入」为 0（无法获取），只统计涨停家数。
    """
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
                    "涨停家数": int(cnt), "来源": "涨停池聚合",
                    "状态": status, "强度分": cnt * 10})
    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out).sort_values("涨停家数", ascending=False).reset_index(drop=True)
    log(f"✅ 涨停池聚合主线: {len(df)} 个板块，TOP3: "
        f"{' | '.join(f'{r.板块}({r.涨停家数}家)' for r in df.head(3).itertuples())}")
    return df


def get_sector_rotation(zt_df=None):
    """
    板块主线监测。
    口径：
      强度分计算（有真实 5日涨幅时）：
        5日涨幅 ≥ 5%   +30
        5日涨幅 ≥ 2%   +20
        5日涨幅 ≥ 0%   +10
        主力净流入 > 0  +30
        主力净流入 < 0  -15
      强度分计算（兜底口径，无真实涨幅时）：
        涨停 ≥ 5 家 → 50 分
        涨停 ≥ 3 家 → 35 分
        涨停 ≥ 2 家 → 20 分
    """
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
        if status == "ok":
            status = "东财行业榜失败"

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
        if status == "ok":
            status = "资金流排名不可用"

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
            if r["5日涨幅"] >= 5:
                score += 30
            elif r["5日涨幅"] >= 2:
                score += 20
            elif r["5日涨幅"] >= 0:
                score += 10
            if r["主力净流入"] > 0:
                score += 30
            elif r["主力净流入"] < 0:
                score -= 15
            return score
        else:
            cnt = r.get("涨停家数", 0)
            if cnt >= 5:
                return 50
            elif cnt >= 3:
                return 35
            elif cnt >= 2:
                return 20
            return 10

    df["强度分"] = df.apply(calc_strength, axis=1)

    def label(r):
        if not has_real_pct:
            cnt = r.get("涨停家数", 0)
            if cnt >= 3:
                return "🔥强势主线"
            elif cnt >= 2:
                return "📈升温中"
            return "—"
        if r["强度分"] >= 60 and r["主力净流入"] > 0:
            return "🔥强势主线"
        if r["强度分"] >= 40:
            return "📈升温中"
        if r["强度分"] < 0 and r["主力净流入"] < 0:
            return "📉走弱"
        if r["5日涨幅"] > 3 and r["主力净流入"] < 0:
            return "⚠️脉冲"
        return "—"

    df["状态"] = df.apply(label, axis=1)
    df = df.sort_values(["强度分", "主力净流入"], ascending=False).reset_index(drop=True)
    log(f"✅ 板块主线分析完成: {len(df)} 个板块 | status={status}")
    return df, status


# ==================== 硬过滤 ====================
def hard_filter(df):
    """
    硬过滤（口径见 HARD_FILTERS）：
      - ST/退市：名称含 'ST' 或 '退'
      - 连板>3：高位接力风险
      - 换手>28%：过度换手
      - 一字板：封板资金 / 流通市值 > 8% 且炸板次数 = 0
      - 总市值<15亿：小盘股易被操纵
      - 尾盘偷袭：首次封板时间 > 14:30
    """
    if df.empty:
        return df, []
    original = len(df)
    reasons = []
    if HARD_FILTERS["st"] and "name" in df.columns:
        mask = df["name"].astype(str).str.contains("ST|退", case=False, na=False)
        if mask.any():
            reasons.append(f"ST/退市: {mask.sum()}只")
            df = df[~mask]
    if "board_count" in df.columns:
        mask = df["board_count"] > HARD_FILTERS["max_boards"]
        if mask.any():
            reasons.append(f"连板>3: {mask.sum()}只")
            df = df[~mask]
    if "turnover" in df.columns:
        mask = df["turnover"] > HARD_FILTERS["max_turnover"]
        if mask.any():
            reasons.append(f"换手>28%: {mask.sum()}只")
            df = df[~mask]
    if HARD_FILTERS["one_word"] and "seal_amount" in df.columns and "circ_market_cap" in df.columns:
        circ = df["circ_market_cap"].replace(0, np.nan)
        ratio = df["seal_amount"] / (circ * 1e8)
        mask = (df["open_times"].fillna(99) == 0) & (ratio > 0.08)
        if mask.any():
            reasons.append(f"一字板: {mask.sum()}只")
            df = df[~mask]
    if "total_market_cap" in df.columns:
        mask = df["total_market_cap"] < HARD_FILTERS["min_market_cap"]
        if mask.any():
            reasons.append(f"市值<15亿: {mask.sum()}只")
            df = df[~mask]
    if HARD_FILTERS["late_afternoon"] and "first_time" in df.columns:
        mask = df["first_time"].apply(lambda x: is_late_afternoon(x))
        if mask.any():
            reasons.append(f"尾盘偷袭: {mask.sum()}只")
            df = df[~mask]
    log(f"🔍 硬过滤: 剔除{original - len(df)}只 ({', '.join(reasons)})")
    return df, reasons


# ==================== 评分 ====================
def score_stock(row, sector_counts, use_fund, use_ma):
    """
    单只股票评分（满分 100）。
    权重模式：
      - 正常模式 WEIGHTS_NORMAL（满分 100）
      - 降级模式 WEIGHTS_DEGRADED（资金流或 K 线熔断时启用）
    所有评分项口径见 WEIGHTS_NORMAL 注释。
    """
    global _fund_circuit_broken, _kline_circuit_broken
    weights = WEIGHTS_DEGRADED if (_fund_circuit_broken or _kline_circuit_broken) else WEIGHTS_NORMAL
    score = 0
    reasons = []

    # 1) 连板数：每层 × weight，上限 15 分
    bc = _to_int(row.get("board_count", 1))
    if bc >= 2:
        s = min(bc * weights["board_count"], 15)
        score += s
        reasons.append(f"连板{bc}层(+{s})")

    # 2) 主线加分：所属行业涨停 ≥ 3 家
    ind = str(row.get("industry", "未知"))
    if sector_counts.get(ind, 0) >= 3:
        s = weights["sector_main"]
        score += s
        reasons.append(f"主线({ind}{sector_counts[ind]}家)(+{s})")

    # 3) 5日/10日主力净流入（仅正常模式）
    if use_fund and weights["fund_flow_5d"] > 0:
        n5, n10 = get_fund_flow(str(row.get("code", "")).zfill(6))
        if n5 > 0:
            s = min(int(n5 / 1000), weights["fund_flow_5d"])
            score += s
            reasons.append(f"5日+{n5:.0f}万(+{s})")
        if n10 > 0:
            s = min(int(n10 / 2000), weights["fund_flow_10d"])
            score += s
            reasons.append(f"10日+{n10:.0f}万(+{s})")

    # 4) 早盘封板：首次封板 < 10:00
    ft = parse_ftime(row.get("first_time"))
    if ft and ft.hour < 10:
        s = weights["morning_lobby"]
        score += s
        reasons.append(f"早盘{ft.strftime('%H:%M')}(+{s})")

    # 5) 未开板：炸板次数 = 0
    if _to_int(row.get("open_times", 1)) == 0:
        s = weights["no_open"]
        score += s
        reasons.append(f"未开板(+{s})")

    # 6) 均线多头 + 回马枪 + 涨停基因（仅正常模式，依赖 K 线）
    if use_ma and not _kline_circuit_broken and weights["ma_bull"] > 0:
        kl = get_kline(str(row.get("code", "")).zfill(6))
        if check_ma_bull(kl):
            s = weights["ma_bull"]
            score += s
            reasons.append(f"均线多头(+{s})")
        if check_return_shot(kl):
            s = weights["return_shot"]
            score += s
            reasons.append(f"回马枪(+{s})")
        if weights["limit_gene"] > 0 and check_limit_gene(kl, 20):
            s = weights["limit_gene"]
            score += s
            reasons.append(f"20日涨停基因(+{s})")

    # 7) 市值加分：总市值 30~100 亿
    mcap = _to_num(row.get("total_market_cap", 0))
    if 30 <= mcap / 1e8 <= 100:
        s = weights["market_cap_bonus"]
        score += s
        reasons.append(f"市值{fmt_mcap(mcap)}(+{s})")

    # 8) 换手健康：5%~15%
    to = _to_num(row.get("turnover", 0))
    if 5 <= to <= 15:
        s = weights["turnover_good"]
        score += s
        reasons.append(f"换手{to:.1f}%健康(+{s})")

    # 9) 封板资金 > 0（v5.8 新增，本地可算）
    if _to_num(row.get("seal_amount", 0)) > 0:
        s = weights["seal_amount"]
        score += s
        reasons.append(f"有封单(+{s})")

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
                log("✅ 企微推送成功")
                return True
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
    msg += f"**阈值**: {min_score_used}分（{'降级模式' if min_score_used != DEFAULT_MIN_SCORE else '正常'}）\n"
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
    log(f"🚀 启动 v5.8 | 模式={mode} | 日期={args.date or '今日(回溯)'}")

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

        # 5. 探针预检（v5.8: 只跑 1 只，且资金熔断后跳过 K 线）
        log(f"🔬 探针预检接口可用性（{PROBE_N} 只）...")
        probe = df.head(min(PROBE_N, len(df)))
        for _, row in probe.iterrows():
            code = str(row.get("code", "")).zfill(6)
            if not args.no_fund:
                get_fund_flow(code)
            if not args.no_ma and not _fund_circuit_broken:
                get_kline(code)

        # 5.5 强制检查熔断
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
        log(f"📝 开始评分 {len(df)} 只（资金={'开' if use_fund else '关'} "
            f"均线={'开' if use_ma else '关'} 阈值={min_score}）...")
        t0 = time.time()
        scores, all_reasons = [], []
        for idx, (_, row) in enumerate(df.iterrows()):
            s, rs = score_stock(row, sector_counts, use_fund=use_fund, use_ma=use_ma)
            scores.append(s)
            all_reasons.append(rs)
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
            cols = [c for c in ["code", "name", "score", "industry", "board_count",
                                 "turnover", "total_market_cap"] if c in df_f.columns]
            df_f[cols].to_csv(
                f"{out_dir}/pick_{datetime.now().strftime('%Y%m%d')}_{mode}.csv",
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
