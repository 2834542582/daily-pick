#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
daily_pick.py —— A股每日选股
【主线】首板/回马枪/均线多头 + 资金过滤
【风控】解禁事件前置过滤（解禁前5天内直接剔除）
数据源：AKShare，无需 token
运行：交易日 16:00 后
"""
import sys
import os
import json
import argparse
import urllib.request
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ================= 参数区 =================
CFG = dict(
    date         = None,
    circ_mv_min  = 20,
    circ_mv_max  = 200,
    turnover_min = 3,
    turnover_max = 12,
    score_min    = 30,
    no_st        = True,
    # 解禁风控
    unlock_days  = 5,        # 解禁前N天内 → 直接剔除
    unlock_max   = 40,       # 解禁占比超N% → 评分扣分
)
# =========================================

def to_num(x):
    try:
        return float(str(x).replace(",", "").replace("%", "").strip())
    except Exception:
        return np.nan

def date_str(d):
    return d.strftime("%Y%m%d")

# ---------------- 解禁风控模块 ----------------
def load_unlock_blacklist(days_ahead=60):
    """
    返回解禁前N天内+解禁占比超标的黑名单 code 集合
    """
    import akshare as ak
    start = datetime.now()
    end = start + timedelta(days=days_ahead)
    try:
        df = ak.stock_restricted_release_detail_em(
            start_date=date_str(start), end_date=date_str(end))
    except Exception:
        return set(), {}

    code_col = [c for c in df.columns if "代码" in c]
    if not code_col:
        return set(), {}
    df["code"] = df[code_col[0]].astype(str).str.zfill(6)

    date_c = [c for c in df.columns if "解禁时间" in c]
    ratio_c = [c for c in df.columns if "占解禁前流通" in c]
    if not date_c or not ratio_c:
        return set(), {}

    df["free_date"] = pd.to_datetime(df[date_c[0]], errors="coerce")
    df["free_ratio"] = df[ratio_c[0]].map(to_num)
    df = df.dropna(subset=["free_date", "free_ratio"])

    today = pd.Timestamp(datetime.now().date())
    df["days"] = (df["free_date"] - today).dt.days

    # 黑名单：解禁前 unlock_days 天内
    black = set(df[df["days"].between(0, CFG["unlock_days"])]["code"])
    # 超额记录：解禁占比>unlock_max，用于扣分
    over = {}
    for _, r in df[df["free_ratio"] > CFG["unlock_max"]].iterrows():
        over[r["code"]] = (round(r["free_ratio"], 1), int(r["days"]))
    return black, over

# ---------------- K线/指标 ----------------
def get_hist(code, lookback=60):
    import akshare as ak
    end = datetime.now()
    start = end - timedelta(days=lookback * 2)
    for fn in [
        lambda: ak.stock_zh_a_hist(symbol=code, period="daily",
                 start_date=date_str(start), end_date=date_str(end), adjust="qfq"),
        lambda: ak.stock_zh_a_daily(symbol=code, start_date=date_str(start),
                 end_date=date_str(end), adjust="qfq"),
    ]:
        try:
            df = fn()
            if df is not None and not df.empty:
                break
        except Exception:
            df = None
    if df is None or df.empty:
        return None
    df = df.rename(columns={"日期":"date","开盘":"open","收盘":"close",
                            "最高":"high","最低":"low","成交量":"vol"})
    df["date"] = pd.to_datetime(df["date"])
    for c in ["open","close","high","low","vol"]:
        if c in df: df[c] = df[c].map(to_num)
    return df.sort_values("date").reset_index(drop=True)

def ma_head(df):
    if df is None or len(df) < 22: return False
    c = df["close"]
    ma5, ma10, ma20 = c.rolling(5).mean().iloc[-1], c.rolling(10).mean().iloc[-1], c.rolling(20).mean().iloc[-1]
    return (ma5 > ma10 > ma20) and (c.iloc[-1] > ma5)

def recent_zt(df, days=20):
    if df is None or len(df) < days: return False
    return df["close"].pct_change().tail(days).max() >= 0.095

def back_attack(df):
    if df is None or len(df) < 12: return False
    c, v = df["close"], df["vol"]
    if v.iloc[-2] <= 0: return False
    return (c.iloc[-2]/c.iloc[-3]-1 >= 0.095) and (v.iloc[-1] < v.iloc[-2]) \
           and (c.iloc[-1] > c.rolling(5).mean().iloc[-1])

def load_flow(code):
    import akshare as ak
    out = {"flow_5d": np.nan, "flow_10d": np.nan}
    try:
        df = ak.stock_individual_fund_flow_rank(indicator="今日")
        r = df[df.iloc[:,0].astype(str).str.zfill(6) == str(code).zfill(6)]
        if not r.empty:
            row = r.iloc[0]
            for s, d in [("5日主力净流入-净额","flow_5d"),("10日主力净流入-净额","flow_10d")]:
                if s in row.index: out[d] = to_num(row[s])
    except Exception:
        pass
    return out

# ---------------- 涨停池 ----------------
def load_zt_pool(date):
    import akshare as ak
    for fn in [lambda: ak.stock_zt_pool_em(date=date),
               lambda: ak.stock_zt_pool(dt=date)]:
        try:
            df = fn()
            if df is not None and not df.empty:
                return df
        except Exception:
            continue
    return None

def normalize_pool(pool):
    if pool is None or pool.empty: return pool
    code_col = [c for c in pool.columns if "代码" in c][0]
    name_col = [c for c in pool.columns if "名称" in c][0]
    pool = pool.rename(columns={code_col:"code", name_col:"name"})
    for col in ["连板数","换手率","流通市值","封单金额","所属行业"]:
        if col not in pool.columns: pool[col] = np.nan
    pool["code"] = pool["code"].astype(str).str.zfill(6)
    for c in ["连板数","换手率","流通市值","封单金额"]:
        pool[c] = pool[c].map(to_num)
    if pool["流通市值"].median(skipna=True) > 1e6:
        pool["流通市值"] = pool["流通市值"] / 1e8
    if CFG["no_st"]:
        pool = pool[~pool["name"].astype(str).str.contains("ST|退", na=False)]
    return pool

# ---------------- 主流程 ----------------
def pick():
    date = CFG["date"] or datetime.now().strftime("%Y%m%d")
    print(f"== 选股日期：{date} | 模式：{CFG['mode']} ==")

    # 解禁风控：先建黑名单
    black, over = load_unlock_blacklist()
    if black:
        print(f"🛡️  解禁风控：{len(black)} 只处于解禁前{CFG['unlock_days']}天窗口，将剔除")

    pool = normalize_pool(load_zt_pool(date))
    if pool is None or pool.empty:
        print("⚠️ 无涨停数据（非交易日/接口限流）")
        return pd.DataFrame()

    n0 = len(pool)
    pool = pool[(pool["流通市值"]>=CFG["circ_mv_min"]) & (pool["流通市值"]<=CFG["circ_mv_max"])]
    pool = pool[(pool["换手率"]>=CFG["turnover_min"]) & (pool["换手率"]<=CFG["turnover_max"])]
    if CFG["mode"] == "first":
        pool = pool[pool["连板数"].fillna(0) <= 1]
    print(f"涨停池 {n0} 只 → 过滤后 {len(pool)} 只")

    results, miss, blocked = [], 0, 0
    for _, row in pool.iterrows():
        code = row["code"]
        if code in black:
            blocked += 1; continue   # 解禁前窗口，直接跳过
        hist = get_hist(code)
        if hist is None or len(hist) < 22:
            miss += 1; continue
        sc, tag = 0, []
        if ma_head(hist):       sc += 30; tag.append("均线多头")
        if recent_zt(hist, 20): sc += 20; tag.append("20日涨停基因")
        if back_attack(hist):   sc += 25; tag.append("回马枪")
        if to_num(row.get("连板数")) >= 2: sc += 15; tag.append("连板惯性")
        if to_num(row.get("封单金额")) > 0: sc += 5; tag.append("有封单")
        # 解禁占比超额扣分
        if code in over:
            sc -= 15; tag.append(f"解禁占比{over[code][0]}%扣分")
        if sc < CFG["score_min"]: continue
        f = load_flow(code)
        if pd.notna(f["flow_5d"]) and f["flow_5d"] > 0: sc += 10; tag.append("5日资金净流入")
        if pd.notna(f["flow_10d"]) and f["flow_10d"] > 0: sc += 15; tag.append("10日资金净流入")
        chg = hist["close"].pct_change().iloc[-1] * 100
        results.append({
            "代码": code, "名称": row["name"], "评分": sc,
            "连板数": int(to_num(row["连板数"])) if pd.notna(row["连板数"]) else 0,
            "换手率%": round(to_num(row["换手率"]),2),
            "流通市值(亿)": round(to_num(row["流通市值"]),1),
            "当日涨幅%": round(chg,2),
            "5日净流入(亿)": round(f["flow_5d"]/1e8,2) if pd.notna(f["flow_5d"]) else np.nan,
            "解禁预警": f"{over[code][0]}%/{over[code][1]}天" if code in over else "无",
            "行业": row.get("所属行业",""),
            "标签": "、".join(tag),
        })

    out = pd.DataFrame(results).sort_values("评分", ascending=False).reset_index(drop=True)
    print(f"K线缺失{miss} | 解禁剔除{blocked} | 达标{len(out)}")
    return out

def industry_rank(date, top=10):
    import akshare as ak
    try:
        pool = normalize_pool(ak.stock_zt_pool_em(date=date))
    except Exception as e:
        print("行业统计失败:", e); return
    if "所属行业" not in pool.columns or pool.empty:
        print("无行业数据"); return
    print("\n【板块涨停家数TOP】")
    for k, v in pool["所属行业"].value_counts().head(top).items():
        if pd.notna(k): print(f"  {k}: {v}家")

# ---------------- 企微推送 ----------------
def push_wecom(csv_path, webhook_url, top_n=15):
    if not webhook_url:
        print("⚠️ 未配置 WECOM_WEBHOOK，跳过推送"); return
    try:
        if not os.path.exists(csv_path) or pd.read_csv(csv_path).empty:
            content = "📭 今日无符合条件标的 —— 空仓也是策略"
        else:
            df = pd.read_csv(csv_path)
            lines = [f"📊 每日选股 {datetime.now().strftime('%m/%d')} | 共{len(df)}只 | 前{top_n}："]
            for _, r in df.head(top_n).iterrows():
                lines.append(
                    f"{r['名称']}({r['代码']}) 评分{r['评分']} "
                    f"| {r['行业']} | {r['换手率%']}%换 | {r['流通市值(亿)']}亿"
                    f"{' | ⚠️'+str(r['解禁预警']) if r.get('解禁预警') not in (None,'无') else ''}"
                )
            lines.append("——量价+资金+解禁风控初筛，非买卖建议")
            content = "\n".join(lines)
        data = json.dumps({"msgtype":"text","text":{"content":content}}, ensure_ascii=False).encode()
        req = urllib.request.Request(webhook_url, data=data,
                                      headers={"Content-Type":"application/json"})
        print("推送:", urllib.request.urlopen(req, timeout=10).read().decode())
    except Exception as e:
        print("推送失败:", e)

# ---------------- 入口 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all", choices=["all","first","back"])
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--date", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--rank", action="store_true")
    ap.add_argument("--no-push", action="store_true", help="只跑不推送")
    a = ap.parse_args()

    CFG["mode"] = a.mode
    if a.date: CFG["date"] = a.date

    out = pick()
    if out.empty:
        print("\n📭 无符合条件标的")
    else:
        pd.set_option("display.unicode.east_asian_width", True)
        pd.set_option("display.max_columns", 14)
        pd.set_option("display.width", 240)
        print("\n" + out.head(a.top).to_string(index=False))
        path = a.out or f"选股结果_{CFG['date'] or datetime.now().strftime('%Y%m%d')}.csv"
        out.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"\n✅ 已保存 {len(out)} 只 → {path}")

    if a.rank:
        industry_rank(CFG["date"] or datetime.now().strftime("%Y%m%d"))

    if not a.no_push:
        wh = os.environ.get("WECOM_WEBHOOK")
        path = a.out or f"选股结果_{CFG['date'] or datetime.now().strftime('%Y%m%d')}.csv"
        if wh and out is not None:
            push_wecom(path, wh)

if __name__ == "__main__":
    main()