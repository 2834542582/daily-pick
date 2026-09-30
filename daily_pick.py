#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每日选股推送 - 主线增强版 v4
修复：资金流向熔断 + 单只超时 + 优雅降级
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
MIN_SCORE = 75
DEFAULT_TOP_N = 15
MAX_BOARDS = 3
MAX_TURNOVER = 28.0
MIN_MARKET_CAP = 15.0
MAIN_BOARD_MIN_STOCKS = 3
RETRY_COUNT = 2
RETRY_DELAY = 3

# 资金流向熔断：连续失败 N 只后全局跳过
FUND_FAIL_THRESHOLD = 3

HARD_FILTERS = {
    "st": True,
    "max_boards": 3,
    "max_turnover": 28.0,
    "one_word": True,
    "min_market_cap": 15.0,
    "late_afternoon": True,
}

SCORE_WEIGHTS = {
    "board_count": 5,
    "sector_main": 20,
    "fund_flow_5d": 15,
    "fund_flow_10d": 10,
    "morning_lobby": 10,
    "no_open": 10,
    "ma_bull": 15,
    "return_shot": 15,
    "market_cap_bonus": 5,
    "turnover_good": 5,
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
                log(f"⚠ 请求失败({i+1}/{RETRY_COUNT}): {e}, {RETRY_DELAY}s后重试", "WARN")
                time.sleep(RETRY_DELAY)
    raise last_error


def _dump_debug(df, label: str):
    try:
        os.makedirs("debug", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if isinstance(df, pd.DataFrame):
            df.head(20).to_csv(f"debug/{label}_{ts}.csv", index=False, encoding="utf-8-sig")
            log(f"🐞 调试落盘: debug/{label}_{ts}.csv | shape={df.shape} | cols={df.columns.tolist()}")
        else:
            with open(f"debug/{label}_{ts}.txt", "w", encoding="utf-8") as f:
                f.write(f"type={type(df)}\nrepr={repr(df)[:500]}\n")
    except Exception as e:
        log(f"⚠ 调试落盘失败: {e}", "WARN")


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
        log(f"⚠ 取交易日历失败: {e}", "WARN")
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
                seen.add(d)
                uniq.append(d)
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


# ==================== 形态判断 ====================
def get_kline(code: str, days: int = 60):
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


# ==================== 资金流向（带熔断）====================
_fund_circuit_broken = False  # 全局熔断标志
_fund_fail_count = 0


def get_fund_flow(code: str):
    global _fund_circuit_broken, _fund_fail_count

    if _fund_circuit_broken:
        return 0.0, 0.0

    try:
        import akshare as ak
        # 单次请求，不重试，快速失败
        df = ak.stock_individual_fund_flow(stock=code)
        if df is None or df.empty:
            _fund_fail_count += 1
            return 0.0, 0.0

        col = "主力净流入"
        if col not in df.columns:
            # 尝试找类似列
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

        # 成功一只，重置失败计数
        _fund_fail_count = 0
        return float(net_5), float(net_10)

    except Exception as e:
        _fund_fail_count += 1
        if _fund_fail_count >= FUND_FAIL_THRESHOLD:
            _fund_circuit_broken = True
            log(f"🚫 资金流向连续失败{_fund_fail_count}只，已熔断跳过后续所有资金请求", "WARN")
        return 0.0, 0.0


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


def score_stock(row, sector_counts, use_fund=True, use_ma=False):
    score = 0
    reasons = []

    bc = _to_int(row.get("board_count", 1))
    if bc >= 2:
        s = min(bc * SCORE_WEIGHTS["board_count"], 15)
        score += s; reasons.append(f"连板{bc}层(+{s})")

    ind = str(row.get("industry", "未知"))
    if sector_counts.get(ind, 0) >= MAIN_BOARD_MIN_STOCKS:
        score += SCORE_WEIGHTS["sector_main"]
        reasons.append(f"主线({ind}{sector_counts[ind]}家)(+{SCORE_WEIGHTS['sector_main']})")

    if use_fund:
        n5, n10 = get_fund_flow(str(row.get("code", "")).zfill(6))
        if n5 > 0:
            s = min(int(n5 / 1000), SCORE_WEIGHTS["fund_flow_5d"])
            score += s; reasons.append(f"5日+{n5:.0f}万(+{s})")
        if n10 > 0:
            s = min(int(n10 / 2000), SCORE_WEIGHTS["fund_flow_10d"])
            score += s; reasons.append(f"10日+{n10:.0f}万(+{s})")

    ft = parse_ftime(row.get("first_time"))
    if ft and ft.hour < 10:
        score += SCORE_WEIGHTS["morning_lobby"]
        reasons.append(f"早盘{ft.strftime('%H:%M')}(+{SCORE_WEIGHTS['morning_lobby']})")

    if _to_int(row.get("open_times", 1)) == 0:
        score += SCORE_WEIGHTS["no_open"]; reasons.append("未开板(+10)")

    if use_ma:
        kl = get_kline(str(row.get("code", "")).zfill(6))
        if check_ma_bull(kl):
            score += SCORE_WEIGHTS["ma_bull"]; reasons.append("均线多头(+15)")
        if check_return_shot(kl):
            score += SCORE_WEIGHTS["return_shot"]; reasons.append("回马枪(+15)")

    mcap = _to_num(row.get("total_market_cap", 0))
    if 30 <= mcap <= 100:
        score += SCORE_WEIGHTS["market_cap_bonus"]; reasons.append(f"市值{mcap:.0f}亿(+5)")

    to = _to_num(row.get("turnover", 0))
    if 5 <= to <= 15:
        score += SCORE_WEIGHTS["turnover_good"]; reasons.append(f"换手{to:.1f}%健康(+5)")

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


def format_message(df, mode, tag, filtered_reasons, fund_broken=False):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    used = df.attrs.get("used_date", "") if hasattr(df, "attrs") else ""
    used_line = f" | 数据日:{used}" if used else ""

    msg = f"## 📈 每日选股推送 - {tag}\n**时间**: {now}{used_line}\n**模式**: {mode}\n"
    if fund_broken:
        msg += "⚠️ *资金流向接口熔断(上游限流)，已跳过资金加分*\n"
    msg += f"**结果**: {len(df)}只 (评分≥{MIN_SCORE})\n\n### 🏆 候选标的\n"

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
    global _fund_circuit_broken

    p = argparse.ArgumentParser(description="每日选股推送")
    p.add_argument("--no-ma", action="store_true", help="跳过均线/回马枪(加速)")
    p.add_argument("--no-fund", action="store_true", help="跳过资金流向")
    p.add_argument("--top", type=int, default=DEFAULT_TOP_N)
    p.add_argument("--date", type=str, default="", help="指定日期YYYYMMDD")
    p.add_argument("--min-score", type=int, default=MIN_SCORE)
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

        df = get_limit_up_pool(args.date)
        if df.empty:
            log("❌ 无涨停数据", "ERROR")
            send_wecom_webhook(os.environ.get("WECOM_WEBHOOK", ""),
                "## ⚠️ 选股未执行\n**时间**: " + datetime.now().strftime("%Y-%m-%d %H:%M") +
                "\n**原因**: 最近交易日涨停池均为空\n**处理**: 已落盘debug/，请查Actions日志")
            sys.exit(1)

        # 板块统计
        sector_counts = {}
        if "industry" in df.columns:
            sector_counts = df["industry"].value_counts().to_dict()
            top3 = sorted(sector_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            log(f"📊 板块TOP3: {' | '.join(f'{k}({v})' for k,v in top3)}")
            main_lines = [k for k, v in top3 if v >= MAIN_BOARD_MIN_STOCKS]
            if main_lines:
                log(f"🔥 主线候选: {main_lines}")

        # 硬过滤
        df, filtered_reasons = hard_filter(df)
        if df.empty:
            log("📭 过滤后无标的")
            send_wecom_webhook(os.environ.get("WECOM_WEBHOOK", ""),
                              format_message(df, mode, tag, filtered_reasons))
            return

        # 评分
        log(f"📝 开始评分 {len(df)} 只...")
        t0 = time.time()
        scores, all_reasons = [], []
        for idx, (_, row) in enumerate(df.iterrows()):
            s, rs = score_stock(row, sector_counts, use_fund=not args.no_fund, use_ma=not args.no_ma)
            scores.append(s); all_reasons.append(rs)
            if (idx + 1) % 10 == 0:
                elapsed = time.time() - t0
                log(f"  ⏳ 已评分 {idx+1}/{len(df)} 只 ({elapsed:.1f}s)")

        df = df.copy()
        df["score"] = scores
        df["reasons"] = all_reasons

        # 过滤+排序+取前N
        df_f = df[df["score"] >= args.min_score].sort_values("score", ascending=False).head(args.top)
        log(f"✅ 达标 {len(df_f)} 只 ≥ {args.min_score}分 (耗时{time.time()-t0:.1f}s)")

        # 落盘
        out_dir = os.environ.get("OUTPUT_DIR", "results")
        os.makedirs(out_dir, exist_ok=True)
        if not df_f.empty:
            cols = [c for c in ["code", "name", "score", "industry", "board_count", "turnover", "total_market_cap"] if c in df_f.columns]
            df_f[cols].to_csv(f"{out_dir}/pick_{datetime.now().strftime('%Y%m%d')}_{mode}.csv",
                              index=False, encoding="utf-8-sig")
            log(f"💾 已保存 {len(df_f)}只")

        # 推送
        send_wecom_webhook(os.environ.get("WECOM_WEBHOOK", ""),
                          format_message(df_f, mode, tag, filtered_reasons,
                                        fund_broken=_fund_circuit_broken))

        log("✅ 选股完成")

    except Exception as e:
        log(f"❌ 异常: {e}", "ERROR")
        traceback.print_exc()
        send_wecom_webhook(os.environ.get("WECOM_WEBHOOK", ""),
            f"## ❌ 脚本异常\n**时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n**错误**: {str(e)[:300]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
