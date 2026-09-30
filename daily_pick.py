#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每日选股推送 - 主线增强版 v5
修复：资金熔断时自动降级（降阈值+权重再分配）+ K线熔断 + 板块主线监测
"""
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
MAX_BOARDS = 3
MAX_TURNOVER = 28.0
MIN_MARKET_CAP = 15.0
MAIN_BOARD_MIN_STOCKS = 3
RETRY_COUNT = 2
RETRY_DELAY = 3
FUND_FAIL_THRESHOLD = 3        # 连续失败N只 → 资金熔断
KLINE_FAIL_THRESHOLD = 5       # 连续失败N只 → K线熔断
DEFAULT_MIN_SCORE = 75         # 正常阈值
DEGRADED_MIN_SCORE = 55        # 降级阈值（资金/K线接口挂了时用）

HARD_FILTERS = {
    "st": True, "max_boards": 3, "max_turnover": 28.0,
    "one_word": True, "min_market_cap": 15.0, "late_afternoon": True,
}

# 权重分两套：正常版 / 降级版
WEIGHTS_NORMAL = {
    "board_count": 5, "sector_main": 20, "fund_flow_5d": 15, "fund_flow_10d": 10,
    "morning_lobby": 10, "no_open": 10, "ma_bull": 15, "return_shot": 15,
    "market_cap_bonus": 5, "turnover_good": 5,
}
# 资金/K线不可用时，把这40分挪给：板块主线、早盘、未开板、换手、市值
WEIGHTS_DEGRADED = {
    "board_count": 5, "sector_main": 35, "fund_flow_5d": 0, "fund_flow_10d": 0,
    "morning_lobby": 18, "no_open": 15, "ma_bull": 0, "return_shot": 0,
    "market_cap_bonus": 12, "turnover_good": 15,
}


# ==================== 基础工具 ====================
def log(msg: str, level: str = "INFO"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


def retry_on_failure(func, *args, **kwargs):
    last_error = None
    for i in range(RETRY_COUNT):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_error = e
            if i < RETRY_COUNT - 1:
                time.sleep(RETRY_DELAY)
    raise last_error


def _dump_debug(df, label: str):
    try:
        os.makedirs("debug", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if isinstance(df, pd.DataFrame):
            df.head(20).to_csv(f"debug/{label}_{ts}.csv", index=False, encoding="utf-8-sig")
            log(f"🐞 落盘: debug/{label}_{ts}.csv | shape={df.shape} | cols={df.columns.tolist()}")
        else:
            with open(f"debug/{label}_{ts}.txt", "w", encoding="utf-8") as f:
                f.write(f"type={type(df)}\nrepr={repr(df)[:500]}\n")
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


# ==================== 涨停池（带回溯）====================
def _get_recent_trade_dates(n: int = 10) -> List[str]:
    try:
        import akshare as ak
        cal = retry_on_failure(ak.tool_trade_date_hist_sina)
        if cal is None or cal.empty or "trade_date" not in cal.columns:
            return []
        cal = cal.copy()
        cal["trade_date"] = pd.to_datetime(cal["trade_date"])
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
        "首次涨停时间": "first_time", "最后涨停时间": "last_time",
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
                seen.add(d); uniq.append(d)
        candidates = uniq
        log(f"📡 获取涨停池: 无参回溯，候选 {candidates[:4]}...")

    for cand in candidates:
        try:
            df = retry_on_failure(ak.stock_zt_pool_em, date=cand)
        except Exception as e:
            log(f"⚠ {cand} 接口异常: {e}", "WARN")
            _dump_debug(pd.DataFrame(), f"ztpool_error_{cand}")
            continue
        if df is None:
            log(f"⚠ {cand} 返回None", "WARN"); _dump_debug(pd.DataFrame(), f"ztpool_none_{cand}"); continue
        if not isinstance(df, pd.DataFrame):
            log(f"⚠ {cand} 类型异常{type(df)}", "WARN"); _dump_debug(df, f"ztpool_type_{cand}"); continue
        if df.empty:
            log(f"⚠ {cand} 空表", "WARN"); _dump_debug(df, f"ztpool_empty_{cand}"); continue
        log(f"✅ 涨停池命中 {cand}: {len(df)} 只")
        _dump_debug(df, f"ztpool_ok_{cand}")
        return _normalize(df.copy(), cand)

    log("❌ 最近交易日均无可涨停数据", "ERROR")
    _dump_debug(pd.DataFrame(), "ztpool_all_failed")
    return pd.DataFrame()


# ==================== 形态判断（带熔断）====================
_kline_circuit_broken = False
_kline_fail_count = 0


def get_kline(code: str, days: int = 60):
    global _kline_circuit_broken, _kline_fail_count
    if _kline_circuit_broken:
        return None
    try:
        import akshare as ak
        end = datetime.now()
        start = end - timedelta(days=days * 2)
        for fn in [
            lambda: ak.stock_zh_a_hist(symbol=code, period="daily",
                     start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"), adjust="qfq"),
            lambda: ak.stock_zh_a_daily(symbol=("sh" if code.startswith("6") else "sz") + code,
                     start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"), adjust="qfq"),
        ]:
            try:
                df = fn()
                if df is not None and not df.empty:
                    if "日期" in df.columns:
                        df["日期"] = pd.to_datetime(df["日期"])
                        df = df.sort_values("日期")
                    return df.tail(days)
            except Exception:
                continue
    except Exception as e:
        log(f"⚠ {code} K线失败: {e}", "WARN")
    _kline_fail_count += 1
    if _kline_fail_count >= KL_INE_FAIL_THRESHOLD := KL_INE_FAIL_THRESHOLD:
        _kline_circuit_broken = True
        log(f"🚫 K线连续失败{_kline_fail_count}只，已熔断跳过后续K线请求", "WARN")
    return None


# 修正：用固定常量
_kline_fail_count = 0
_kline_circuit_broken = False


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


# ==================== 资金流向（带熔断）====================
_fund_circuit_broken = False
_fund_fail_count = 0


def get_fund_flow(code: str):
    global _fund_circuit_broken, _fund_fail_count
    if _fund_circuit_broken:
        return 0.0, 0.0
    try:
        import akshare as ak
        df = ak.stock_individual_fund_flow(stock=code)
        if df is None or df.empty:
            _fund_fail_count += 1
            return 0.0, 0.0
        col = "主力净流入"
        if col not in df.columns:
            for c in df.columns:
                if "主力" in str(c) and "净" in str(c):
                    col = c
                    break
            else:
                _fund_fail_count += 1
                return 0.0, 0.0
        df = df.sort_values(df.columns[0], ascending=False)
        net_5 = pd.to_numeric(df[col].head(5), errors="coerce").sum()
        net_10 = pd.to_numeric(df[col].head(10), errors="coerce").sum()
        _fund_fail_count = 0
        return float(net_5), float(net_10)
    except Exception:
        _fund_fail_count += 1
        if _fund_fail_count >= FUND_FAIL_THRESHOLD:
            _fund_circuit_broken = True
            log(f"🚫 资金流向连续失败{_fund_fail_count}只，已熔断", "WARN")
        return 0.0, 0.0


# ==================== 板块主线监测 ====================
def get_sector_rotation() -> Tuple[pd.DataFrame, str]:
    """
    返回板块主线 DataFrame + 状态说明。
    列: 板块 | 5日涨幅 | 5日主力净流入 | 强度分 | 状态
    状态: 强势主线 / 升温中 / 走弱 / 脉冲
    """
    import akshare as ak
    rows = []
    market_idx_pct = 0.0
    status = "ok"

    try:
        # 大盘基准
        m = retry_on_failure(ak.stock_market_fund_flow)
        if m is not None and not m.empty and "今日涨跌幅" in m.columns:
            market_idx_pct = _to_num(m.iloc[0]["今日涨跌幅"])
    except Exception as e:
        log(f"⚠ 大盘基准获取失败: {e}", "WARN")
        status = "大盘数据缺失，相对强度未计算"

    try:
        # 行业涨幅榜
        ind = retry_on_failure(ak.stock_board_industry_name_em)
        if ind is not None and not ind.empty:
            name_col = next((c for c in ind.columns if "名称" in c), ind.columns[0])
            pct_col = next((c for c in ind.columns if "涨跌幅" in c), None)
            for _, r in ind.iterrows():
                rows.append({"板块": str(r[name_col]), "类型": "行业",
                             "5日涨幅": _to_num(r.get(pct_col, 0)) if pct_col else 0.0,
                             "主力净流入": 0.0})
    except Exception as e:
        log(f"⚠ 行业涨幅榜失败: {e}", "WARN")
        status = "行业涨幅榜失败"

    try:
        # 行业资金流（5日）
        flow = ak.stock_sector_fund_flow_rank(indicator="5日", sector_type="行业资金流向")
        if flow is not None and not flow.empty:
            name_col = next((c for c in flow.columns if "名称" in c or "板块" in c), flow.columns[0])
            net_col = next((c for c in flow.columns if "主力" in c and ("净额" in c or "净流入" in c)), None)
            pct_col = next((c for c in flow.columns if "涨跌幅" in c), None)
            for _, r in flow.iterrows():
                nm = str(r[name_col])
                net = _to_num(r.get(net_col, 0)) if net_col else 0.0
                pct = _to_num(r.get(pct_col, 0)) if pct_col else 0.0
                # 合并到 rows
                found = False
                for row in rows:
                    if row["板块"] == nm:
                        row["主力净流入"] = net
                        if pct_col:
                            row["5日涨幅"] = pct
                        found = True
                        break
                if not found:
                    rows.append({"板块": nm, "类型": "行业", "5日涨幅": pct, "主力净流入": net})
    except Exception as e:
        log(f"⚠ 行业资金流失败: {e}", "WARN")
        status = "资金流接口不可用，仅按涨幅排名"

    if not rows:
        return pd.DataFrame(), "无板块数据"

    df = pd.DataFrame(rows).drop_duplicates(subset=["板块"]).copy()

    # 计算强度分
    def calc_strength(r):
        score = 0
        if r["5日涨幅"] >= 5:      score += 30
        elif r["5日涨幅"] >= 2:    score += 20
        elif r["5日涨幅"] >= 0:    score += 10
        else:                     score += 0
        if r["主力净流入"] > 0:    score += 30
        elif r["主力净流入"] < 0:  score -= 15
        # 相对大盘
        excess = r["5日涨幅"] - market_idx_pct
        if excess >= 3:           score += 20
        elif excess >= 1:         score += 10
        elif excess < -1:         score -= 10
        return score

    df["强度分"] = df.apply(calc_strength, axis=1)

    def label(r):
        if r["强度分"] >= 60 and r["主力净流入"] > 0:  return "🔥强势主线"
        if r["强度分"] >= 40:                          return "📈升温中"
        if r["强度分"] < 0 and r["主力净流入"] < 0:    return "📉走弱"
        if r["5日涨幅"] > 3 and r["主力净流入"] < 0:   return "⚠️脉冲"
        return "—"
    df["状态"] = df.apply(label, axis=1)
    df = df.sort_values("强度分", ascending=False).reset_index(drop=True)
    return df, status


# ==================== 过滤 & 评分 ====================
def hard_filter(df: pd.DataFrame):
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
    if sector_counts.get(ind, 0) >= MAIN_BOARD_MIN_STOCKS:
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
    if 30 <= mcap <= 100:
        s = weights["market_cap_bonus"]; score += s; reasons.append(f"市值{mcap:.0f}亿(+{s})")

    to = _to_num(row.get("turnover", 0))
    if 5 <= to <= 15:
        s = weights["turnover_good"]; score += s; reasons.append(f"换手{to:.1f}%健康(+{s})")

    return min(score, 100), reasons


# ==================== 企微推送 ====================
def send_wecom_webhook(webhook_url: str, content: str) -> bool:
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


def format_sector_section(sec_df: pd.DataFrame, status: str) -> str:
    if sec_df is None or sec_df.empty:
        return ""
    msg = "\n### 🌐 板块主线监测\n"
    if status:
        msg += f"> {status}\n"
    # 强势主线
    strong = sec_df[sec_df["状态"] == "🔥强势主线"].head(5)
    if not strong.empty:
        msg += "\n**🔥 强势主线**\n"
        for _, r in strong.iterrows():
            msg += f"- {r['板块']} | 5日{r['5日涨幅']:+.1f}% | 资金{r['主力净流入']/1e8:+.2f}亿\n"
    # 走弱
    weak = sec_df[sec_df["状态"] == "📉走弱"].head(3)
    if not weak.empty:
        msg += "\n**📉 转弱方向**\n"
        for _, r in weak.iterrows():
            msg += f"- {r['板块']} | 5日{r['5日涨幅']:+.1f}% | 资金{r['主力净流入']/1e8:+.2f}亿\n"
    pulse = sec_df[sec_df["状态"] == "⚠️脉冲"].head(3)
    if not pulse.empty:
        msg += "\n**⚠️ 脉冲（不追）**\n"
        for _, r in pulse.iterrows():
            msg += f"- {r['板块']} | 涨但资金{r['主力净流入']/1e8:+.2f}亿(量价背离)\n"
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
            msg += f"- 换手:{_to_num(r.get('turnover',0)):.1f}% | 市值:{_to_num(r.get('total_market_cap',0)):.0f}亿\n"
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

    p = argparse.ArgumentParser(description="每日选股推送")
    p.add_argument("--no-ma", action="store_true")
    p.add_argument("--no-fund", action="store_true")
    p.add_argument("--top", type=int, default=DEFAULT_TOP_N)
    p.add_argument("--date", type=str, default="")
    p.add_argument("--min-score", type=int, default=None, help="手动指定阈值，不指定则自动")
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
    log(f"🚀 启动 | 模式={mode} | 日期={args.date or '今日(回溯)'}")

    try:
        import urllib.request

        # 阈值自动决策
        min_score = args.min_score if args.min_score is not None else DEFAULT_MIN_SCORE

        # 1. 板块主线监测（先做，不受个股接口影响）
        log("🌐 板块主线监测...")
        sec_df, sec_status = get_sector_rotation()
        if not sec_df.empty:
            log(f"✅ 板块数据 {len(sec_df)} 个，TOP3: {' | '.join(f\"{r['板块']}({r['状态']})\" for _, r in sec_df.head(3).iterrows())}")

        # 2. 涨停池
        df = get_limit_up_pool(args.date)
        if df.empty:
            log("❌ 无涨停数据", "ERROR")
            send_wecom_webhook(os.environ.get("WECOM_WEBHOOK", ""),
                "## ⚠️ 选股未执行\n**时间**: " + datetime.now().strftime("%Y-%m-%d %H:%M") +
                "\n**原因**: 最近交易日涨停池均为空\n**处理**: 已落盘debug/，请查Actions日志")
            sys.exit(1)

        # 3. 板块统计
        sector_counts = {}
        if "industry" in df.columns:
            sector_counts = df["industry"].value_counts().to_dict()
            top3 = sorted(sector_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            log(f"📊 涨停板块TOP3: {' | '.join(f'{k}({v})' for k,v in top3)}")
            main_lines = [k for k, v in top3 if v >= MAIN_BOARD_MIN_STOCKS]
            if main_lines:
                log(f"🔥 主线候选: {main_lines}")

        # 4. 硬过滤
        df, filtered_reasons = hard_filter(df)
        if df.empty:
            log("📭 过滤后无标的")
            send_wecom_webhook(os.environ.get("WECOM_WEBHOOK", ""),
                              format_message(df, mode, tag, filtered_reasons, sec_df, sec_status, min_score))
            return

        # 5. 自动降级判断
        use_fund = not args.no_fund and not _fund_circuit_broken
        use_ma = not args.no_ma and not _kline_circuit_broken
        if (_fund_circuit_broken or _kline_circuit_broken) and args.min_score is None:
            min_score = DEGRADED_MIN_SCORE
            log(f"⚠️ 接口降级，阈值自动从{DEFAULT_MIN_SCORE}调整为{min_score}", "WARN")
            if _fund_circuit_broken:
                log("   → 资金维度权重已挪给：板块主线/早盘/未开板/换手/市值", "WARN")
            if _kline_circuit_broken:
                log("   → 均线/回马枪维度已关闭", "WARN")

        # 6. 评分
        log(f"📝 开始评分 {len(df)} 只（资金={'开' if use_fund else '关'} 均线={'开' if use_ma else '关'}）...")
        t0 = time.time()
        scores, all_reasons = [], []
        for idx, (_, row) in enumerate(df.iterrows()):
            s, rs = score_stock(row, sector_counts, use_fund=use_fund, use_ma=use_ma)
            scores.append(s); all_reasons.append(rs)
            if (idx + 1) % 10 == 0:
                log(f"  ⏳ {idx+1}/{len(df)} 只 ({time.time()-t0:.1f}s)")

        df = df.copy()
        df["score"] = scores
        df["reasons"] = all_reasons
        df_f = df[df["score"] >= min_score].sort_values("score", ascending=False).head(args.top)
        log(f"✅ 达标 {len(df_f)} 只 ≥ {min_score}分 ({time.time()-t0:.1f}s)")

        # 7. 落盘
        out_dir = os.environ.get("OUTPUT_DIR", "results")
        os.makedirs(out_dir, exist_ok=True)
        if not df_f.empty:
            cols = [c for c in ["code", "name", "score", "industry", "board_count", "turnover", "total_market_cap"] if c in df_f.columns]
            df_f[cols].to_csv(f"{out_dir}/pick_{datetime.now().strftime('%Y%m%d')}_{mode}.csv",
                              index=False, encoding="utf-8-sig")
            log(f"💾 已保存 {len(df_f)}只")

        # 8. 推送
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
