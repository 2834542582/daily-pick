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
    "kline_lookback": 30,  # 拉最近N天，取最后一根
}

def get_trade_date():
    if CFG["date"]:
        return CFG["date"]
    return datetime.datetime.now().strftime("%Y%m%d")

def _to_num(x):
    try:
        return float(str(x).replace(",", "").strip())
    except Exception:
        return 0.0

def get_limit_up_pool(date_str):
    try:
        df = ak.stock_zt_pool_em(date=date_str)
        return df
    except Exception as e:
        print(f"⚠️ 获取涨停池失败: {e}")
        return pd.DataFrame()

def get_kline(code):
    """拉最近N天日线，返回最新一根（Series）或 None"""
    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=CFG["kline_lookback"] * 2)
    for fn in [
        lambda: ak.stock_zh_a_hist(symbol=code, period="daily",
                 start_date=start.strftime("%Y%m%d"),
                 end_date=end.strftime("%Y%m%d"), adjust="qfq"),
        lambda: ak.stock_zh_a_daily(symbol=code,
                 start_date=start.strftime("%Y%m%d"),
                 end_date=end.strftime("%Y%m%d"), adjust="qfq"),
    ]:
        try:
            df = fn()
            if df is not None and not df.empty:
                df = df.sort_values("日期" if "日期" in df.columns else df.columns[0])
                return df.iloc[-1]
        except Exception:
            continue
    return None

def load_unlock_ratios():
    """返回 {code: 解禁比例}"""
    ratios = {}
    try:
        df = ak.stock_restricted_release_queue_em()
    except Exception as e:
        print(f"⚠️ 解禁接口失败: {e}")
        return ratios
    if df is None or df.empty:
        return ratios
    # 自适应找代码列和解禁比例列
    code_col = next((c for c in df.columns if "代码" in c), None)
    ratio_col = next((c for c in df.columns if "比例" in c or "占比" in c), None)
    if not code_col:
        return ratios
    for _, r in df.iterrows():
        ratios[str(r[code_col]).zfill(6)] = _to_num(r.get(ratio_col, 0))
    return ratios

def calc_score(row_dict, unlock_ratio=0.0):
    score = 50
    turnover = _to_num(row_dict.get("换手率", 0))
    if turnover > 15:
        score += 12
    elif turnover > 8:
        score += 8
    elif turnover > 3:
        score += 4

    amount = _to_num(row_dict.get("成交额", 0))
    if amount > 20e8:
        score += 15
    elif amount > 10e8:
        score += 10
    elif amount > 3e8:
        score += 5

    change = _to_num(row_dict.get("涨跌幅", 0))
    if change >= 9.8:
        score += 10

    circ_mv = _to_num(row_dict.get("流通市值", 0))
    if 20e8 < circ_mv < 100e8:
        score += 8

    if unlock_ratio > CFG["unlock_max"]:
        score -= 15
    return max(0, min(score, 100))

def pick():
    date_str = get_trade_date()
    print(f"\n== 选股日期：{date_str} | 模式：first ==")

    zt_df = get_limit_up_pool(date_str)
    if zt_df is None or zt_df.empty:
        print("📭 涨停池为空（非交易日/接口限流/数据未更新）")
        return pd.DataFrame()
    print(f"涨停池 {len(zt_df)} 只")

    unlock_ratios = load_unlock_ratios()
    if unlock_ratios:
        print(f"🛡️ 解禁池 {len(unlock_ratios)} 只，将用于扣分")
    else:
        print("ℹ️ 解禁数据未取到，跳过解禁风控")

    # 自适应找涨停池字段
    code_col = next((c for c in zt_df.columns if "代码" in c), zt_df.columns[0])
    name_col = next((c for c in zt_df.columns if "名称" in c), zt_df.columns[1])

    results, blocked, miss = [], 0, 0

    for _, row in zt_df.iterrows():
        code = str(row.get(code_col, "")).zfill(6)
        name = str(row.get(name_col, ""))

        if "ST" in name.upper():
            blocked += 1
            continue

        bar = get_kline(code)
        if bar is None:
            miss += 1
            continue

        unlock_ratio = unlock_ratios.get(code, 0.0)

        row_dict = {
            "换手率": row.get("换手率", 0),
            "成交额": row.get("成交额", 0),
            "涨跌幅": row.get("涨跌幅", 0),
            "流通市值": row.get("流通市值", 0),
        }

        score = calc_score(row_dict, unlock_ratio)
        if score < CFG["min_score"]:
            continue

        results.append({
            "代码": code,
            "名称": name,
            "评分": score,
            "换手率": _to_num(row_dict["换手率"]),
            "成交额_亿": round(_to_num(row_dict["成交额"]) / 1e8, 1),
            "涨跌幅": _to_num(row_dict["涨跌幅"]),
            "解禁占比": unlock_ratio,
        })

    print(f"达标 {len(results)} 只 | 解禁剔除{blocked} | K线缺失{miss}")

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
                f"{_to_num(row.get('换手率')):.1f}%换 | "
                f"{_to_num(row.get('成交额_亿')):.1f}亿"
                + (f" | ⚠️解禁{_to_num(row.get('解禁占比')):.0f}%" if _to_num(row.get('解禁占比')) > 0 else "")
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
