import os
import sys
import argparse
import datetime
import json
import urllib.request
import pandas as pd
import numpy as np
import akshare as ak

CFG = {
    "date": None,
    "top_n": 15,
    "unlock_days": 5,
    "unlock_max": 40,
    "min_score": 30,
}

def get_trade_date():
    if CFG["date"]:
        return CFG["date"]
    return datetime.datetime.now().strftime("%Y%m%d")

def get_limit_up_pool(date_str):
    try:
        df = ak.stock_zt_pool_em(date=date_str)
        return df
    except Exception as e:
        print(f"⚠️ 获取涨停池失败: {e}")
        return pd.DataFrame()

def get_unlock_calendar():
    try:
        df = ak.stock_restricted_release_queue_em()
        return df
    except Exception as e:
        print(f"⚠️ 解禁接口失败: {e}")
        return pd.DataFrame()

def calc_score(row, unlock_ratio=0):
    score = 50
    turnover = row.get("换手率", 0) or 0
    if turnover > 15:
        score += 12
    elif turnover > 8:
        score += 8
    elif turnover > 3:
        score += 4
    amount = row.get("成交额", 0) or 0
    if amount > 20e8:
        score += 15
    elif amount > 10e8:
        score += 10
    elif amount > 3e8:
        score += 5
    change = row.get("涨跌幅", 0) or 0
    if change >= 9.8:
        score += 10
    circ_mv = row.get("流通市值", 0) or 0
    if 20e8 < circ_mv < 100e8:
        score += 8
    if unlock_ratio > CFG["unlock_max"]:
        score -= 15
    return max(0, min(score, 100))

def pick():
    date_str = get_trade_date()
    print(f"\n== 选股日期：{date_str} | 模式：first ==")

    zt_df = get_limit_up_pool(date_str)
    if zt_df.empty:
        print("📭 涨停池为空（非交易日/接口限流）")
        return pd.DataFrame()

    print(f"涨停池 {len(zt_df)} 只")

    unlock_df = get_unlock_calendar()
    unlock_set = set()
    if not unlock_df.empty and "代码" in unlock_df.columns:
        try:
            recent = unlock_df.copy()
            recent["解禁日期"] = pd.to_datetime(recent.get("解禁日期", ""), errors="coerce")
            cutoff = pd.Timestamp.now() + pd.Timedelta(days=CFG["unlock_days"])
            mask = (recent["解禁日期"] <= cutoff) & (recent["解禁日期"] >= pd.Timestamp.now())
            unlock_set = set(recent.loc[mask, "代码"].astype(str).tolist())
        except Exception as e:
            print(f"⚠️ 解禁数据处理异常: {e}")

    if unlock_set:
        print(f"🛡️ 解禁风控：{len(unlock_set)} 只处于解禁前{CFG['unlock_days']}天窗口，将剔除")

    results = []
    blocked = 0
    miss = 0

    for _, row in zt_df.iterrows():
        code = str(row.get("代码", "")).zfill(6)
        name = row.get("名称", "")

        if code in unlock_set:
            blocked += 1
            continue

        try:
            kline = ak.stock_zh_a_hist(symbol=code, period="daily",
                                        start_date=date_str, end_date=date_str, adjust="qfq")
            if kline is None or kline.empty:
                miss += 1
                continue
        except Exception:
            miss += 1
            continue

        unlock_ratio = 0
        if not unlock_df.empty:
            try:
                codes = unlock_df.get("代码", pd.Series()).astype(str).tolist()
                if code in codes:
                    idx = codes.index(code)
                    unlock_ratio = float(unlock_df.iloc[idx].get("解禁比例", 0) or 0)
            except Exception:
                pass

        score = calc_score(row, unlock_ratio)
        if score < CFG["min_score"]:
            continue

        results.append({
            "代码": code,
            "名称": name,
            "评分": score,
            "换手率": row.get("换手率", 0),
            "成交额_亿": round((row.get("成交额", 0) or 0) / 1e8, 1),
            "涨跌幅": row.get("涨跌幅", 0),
        })

    print(f"过滤后 {len(results)} 只 | 解禁剔除{blocked} | K线缺失{miss}")

    if not results:
        print("📭 今日无符合条件标的 —— 空仓也是策略")
        return pd.DataFrame()

    out = pd.DataFrame(results).sort_values("评分", ascending=False).reset_index(drop=True)
    print(f"达标 {len(out)} 只，已按评分排序")
    return out

def push_wecom(df, webhook):
    if df is None or df.empty:
        content = "📊 每日选股 | 今日无符合条件标的\n空仓也是策略 🛡️"
    else:
        top = df.head(CFG["top_n"])
        lines = [f"📊 每日选股 {get_trade_date()} | 共{len(df)}只 | 前{CFG['top_n']}："]
        for _, row in top.iterrows():
            lines.append(
                f"{row['名称']}({row['代码']}) 评分{row['评分']} | "
                f"{row.get('换手率', 0):.1f}%换 | {row.get('成交额_亿', 0):.1f}亿"
            )
        content = "\n".join(lines)

    data = {
        "msgtype": "text",
        "text": {"content": content}
    }

    req = urllib.request.Request(
        webhook,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = resp.read().decode("utf-8")
            print(f"推送结果: {result}")
    except Exception as e:
        print(f"推送失败: {e}")

def main():
    parser = argparse.ArgumentParser(description="每日选股推送")
    parser.add_argument("--mode", default="first", help="选股模式")
    parser.add_argument("--top", type=int, default=15, help="推送前N只")
    parser.add_argument("--rank", action="store_true", help="按评分排序")
    parser.add_argument("--no-push", action="store_true", help="不推送")
    args = parser.parse_args()

    CFG["top_n"] = args.top

    out = pick()

    if out is not None and not out.empty:
        print(f"\n📋 选股结果预览：")
        print(out.head(CFG["top_n"]).to_string(index=False))
    else:
        print("\n📭 无符合条件标的，跳过输出")

    if not args.no_push:
        wh = os.environ.get("WECOM_WEBHOOK")
        if wh:
            push_wecom(out, wh)
        else:
            print("⚠️ WECOM_WEBHOOK 未配置，跳过推送")

    return 0

if __name__ == "__main__":
    sys.exit(main())
