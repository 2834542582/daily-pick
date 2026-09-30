import os
import sys
import argparse
import datetime
import json
import urllib.request
import time
import pandas as pd
import numpy as np
import akshare as ak

CFG = {
    "date": None,
    "top_n": 15,
    "min_score": 30,
}

def get_trade_date():
    if CFG["date"]:
        return CFG["date"]
    return datetime.datetime.now().strftime("%Y%m%d")

def _to_num(x):
    try:
        return float(str(x).replace(",", "").replace("%", "").strip())
    except Exception:
        return 0.0

def get_limit_up_pool(date_str):
    """获取涨停池，带重试"""
    for attempt in range(3):
        try:
            df = ak.stock_zt_pool_em(date=date_str)
            if df is not None and not df.empty:
                return df
            print(f"⚠️ 涨停池为空（尝试 {attempt+1}/3）")
        except Exception as e:
            print(f"⚠️ 获取涨停池失败（尝试 {attempt+1}/3）: {e}")
        if attempt < 2:
            time.sleep(5)
    return pd.DataFrame()

def load_unlock_set():
    """获取解禁股票代码集合"""
    for attempt in range(2):
        try:
            df = ak.stock_restricted_release_queue_em()
            if df is not None and not df.empty:
                # 自适应找代码列
                code_col = next((c for c in df.columns if "代码" in c or "code" in c.lower()), df.columns[0])
                return set(str(x).zfill(6) for x in df[code_col].tolist())
        except Exception as e:
            print(f"⚠️ 解禁接口失败（尝试 {attempt+1}/2）: {e}")
        if attempt < 1:
            time.sleep(3)
    return set()

def calc_score(row_dict):
    """基于涨停池字段计算评分"""
    score = 40  # 基础分：能涨停本身就是强势信号

    turnover = _to_num(row_dict.get("换手率", 0))
    if 5 <= turnover <= 15:
        score += 20
    elif 15 < turnover <= 25:
        score += 10
    elif turnover > 25:
        score += 5   # 换手太高，有出货嫌疑

    amount = _to_num(row_dict.get("成交额", 0))
    if amount >= 10e8:
        score += 15
    elif amount >= 5e8:
        score += 10
    elif amount >= 2e8:
        score += 5

    circ_mv = _to_num(row_dict.get("流通市值", 0))
    if 30e8 <= circ_mv <= 200e8:
        score += 15
    elif 200e8 < circ_mv <= 500e8:
        score += 8

    change = _to_num(row_dict.get("涨跌幅", 0))
    if change >= 10:
        score += 10   # 强势封死

    return max(0, min(score, 100))

def pick():
    date_str = get_trade_date()
    print(f"\n== 选股日期：{date_str} | 模式：first ==")

    zt_df = get_limit_up_pool(date_str)
    if zt_df is None or zt_df.empty:
        print("📭 涨停池为空（非交易日/接口持续失败）")
        return pd.DataFrame()

    print(f"涨停池 {len(zt_df)} 只")

    # 解禁数据（可选风控）
    unlock_set = load_unlock_set()
    if unlock_set:
        print(f"🛡️ 解禁池 {len(unlock_set)} 只")
    else:
        print("ℹ️ 解禁数据未取到，跳过解禁风控")

    # 自适应找列名
    cols = zt_df.columns.tolist()
    code_col = next((c for c in cols if "代码" in c), cols[0])
    name_col = next((c for c in cols if "名称" in c), cols[1] if len(cols) > 1 else cols[0])

    print(f"字段: 代码='{code_col}', 名称='{name_col}'")
    print(f"可用字段: {cols[:10]}...")

    results = []
    blocked = 0

    for _, row in zt_df.iterrows():
        code = str(row.get(code_col, "")).zfill(6)
        name = str(row.get(name_col, ""))

        # 剔除 ST
        if "ST" in name.upper():
            blocked += 1
            continue

        # 剔除解禁股
        if code in unlock_set:
            blocked += 1
            continue

        # 直接用涨停池字段评分（不再拉K线）
        row_dict = {
            "换手率": row.get("换手率", 0),
            "成交额": row.get("成交额", 0),
            "涨跌幅": row.get("涨跌幅", 0),
            "流通市值": row.get("流通市值", 0),
        }

        score = calc_score(row_dict)
        if score < CFG["min_score"]:
            continue

        results.append({
            "代码": code,
            "名称": name,
            "评分": score,
            "换手率": _to_num(row_dict["换手率"]),
            "成交额_亿": round(_to_num(row_dict["成交额"]) / 1e8, 1),
            "涨跌幅": _to_num(row_dict["涨跌幅"]),
            "流通市值_亿": round(_to_num(row_dict["流通市值"]) / 1e8, 1),
        })

    print(f"达标 {len(results)} 只 | 剔除{blocked}")

    if not results:
        print("📭 今日无符合条件标的 —— 空仓也是策略")
        return pd.DataFrame()

    out = pd.DataFrame(results).sort_values("评分", ascending=False).reset_index(drop=True)
    print(f"✅ 最终输出 {len(out)} 只，已按评分排序")
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
                f"{_to_num(row.get('换手率')):.1f}%换 | "
                f"{_to_num(row.get('成交额_亿')):.1f}亿 | "
                f"{_to_num(row.get('流通市值_亿')):.0f}亿流通"
            )
        content = "\n".join(lines)

    data = {"msgtype": "text", "text": {"content": content}}
    req = urllib.request.Request(
        webhook, data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            print(f"推送结果: {resp.read().decode('utf-8')}")
    except Exception as e:
        print(f"推送失败: {e}")

def main():
    parser = argparse.ArgumentParser(description="每日选股推送")
    parser.add_argument("--mode", default="first")
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--rank", action="store_true")
    parser.add_argument("--no-push", action="store_true")
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
