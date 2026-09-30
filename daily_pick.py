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
    "min_score": 60,          # 门槛从30提到60
    "max_board_days": 3,      # 连板天数上限（>3板剔除）
    "max_turnover": 28.0,     # 换手率上限（%）
    "fengban_ratio": 0.05,    # 封板资金/流通市值上限（一字板剔除）
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

def _to_int(x):
    try:
        return int(float(str(x).replace(",", "").strip()))
    except Exception:
        return 1

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
                code_col = next((c for c in df.columns if "代码" in c or "code" in c.lower()), df.columns[0])
                return set(str(x).zfill(6) for x in df[code_col].tolist())
        except Exception as e:
            print(f"⚠️ 解禁接口失败（尝试 {attempt+1}/2）: {e}")
        if attempt < 1:
            time.sleep(3)
    return set()

def get_board_map(date_str):
    """
    返回 {板块名称: 涨停只数}，用于板块效应加分。
    取当日行业板块涨幅榜前10，作为强势板块参考。
    """
    board_map = {}
    try:
        df = ak.stock_board_industry_name_em()
        if df is not None and not df.empty:
            top = df.head(10)
            name_col = next((c for c in top.columns if "名称" in c), top.columns[0])
            for _, r in top.iterrows():
                board_map[str(r[name_col])] = board_map.get(str(r[name_col]), 0) + 1
    except Exception as e:
        print(f"⚠️ 板块数据未取到: {e}")
    return board_map

def calc_score(row_dict, board_map):
    """基于涨停池字段计算评分"""
    score = 40  # 基础分：涨停本身就是强势信号

    # 换手率（5-15最佳，>25警惕出货）
    turnover = _to_num(row_dict.get("换手率", 0))
    if 5 <= turnover <= 15:
        score += 20
    elif 15 < turnover <= 20:
        score += 12
    elif 20 < turnover <= CFG["max_turnover"]:
        score += 5
    elif turnover > CFG["max_turnover"]:
        score -= 10   # 换手爆表扣分

    # 成交额
    amount = _to_num(row_dict.get("成交额", 0))
    if amount >= 10e8:
        score += 15
    elif amount >= 5e8:
        score += 10
    elif amount >= 2e8:
        score += 5

    # 流通市值（30-200亿最佳弹性）
    circ_mv = _to_num(row_dict.get("流通市值", 0))
    if 30e8 <= circ_mv <= 200e8:
        score += 15
    elif 200e8 < circ_mv <= 500e8:
        score += 8
    elif circ_mv > 500e8:
        score += 3    # 大市值弹性差

    # 涨跌幅（20cm加分，弹性更大）
    change = _to_num(row_dict.get("涨跌幅", 0))
    if change >= 19.9:
        score += 12
    elif change >= 9.9:
        score += 8

    # 首板加分（比高位接力安全）
    lb_days = _to_int(row_dict.get("连板天数", 1))
    if lb_days == 1:
        score += 10
    elif lb_days == 2:
        score += 5

    # 早盘封板加分（10:30前封板=主力决心强）
    fengban_time = str(row_dict.get("首次封板时间", ""))
    if fengban_time and ("09:" in fengban_time or "10:" in fengban_time):
        if fengban_time <= "10:30":
            score += 10
        elif fengban_time <= "11:30":
            score += 5

    # 板块效应加分
    board_name = str(row_dict.get("所属行业", ""))
    if board_name and board_name in board_map:
        score += 8

    return max(0, min(score, 100))

def pick():
    date_str = get_trade_date()
    print(f"\n== 选股日期：{date_str} | 模式：first ==")

    zt_df = get_limit_up_pool(date_str)
    if zt_df is None or zt_df.empty:
        print("📭 涨停池为空（非交易日/接口持续失败）")
        return pd.DataFrame()
    print(f"涨停池 {len(zt_df)} 只")

    # 解禁数据
    unlock_set = load_unlock_set()
    if unlock_set:
        print(f"🛡️ 解禁池 {len(unlock_set)} 只")
    else:
        print("ℹ️ 解禁数据未取到，跳过解禁风控")

    # 板块数据
    board_map = get_board_map(date_str)
    if board_map:
        print(f"🔥 强势板块 {len(board_map)} 个")
    else:
        print("ℹ️ 板块数据未取到，跳过板块加分")

    # 自适应找列名
    cols = zt_df.columns.tolist()
    code_col = next((c for c in cols if "代码" in c), cols[0])
    name_col = next((c for c in cols if "名称" in c), cols[1] if len(cols) > 1 else cols[0])

    print(f"字段: 代码='{code_col}', 名称='{name_col}'")
    print(f"可用字段: {cols}")

    results = []
    blocked = 0
    reason = {"连板过高": 0, "换手过高": 0, "一字板": 0, "ST": 0, "解禁": 0}

    for _, row in zt_df.iterrows():
        code = str(row.get(code_col, "")).zfill(6)
        name = str(row.get(name_col, ""))

        # 剔除 ST
        if "ST" in name.upper():
            reason["ST"] += 1
            blocked += 1
            continue

        # 剔除解禁股
        if code in unlock_set:
            reason["解禁"] += 1
            blocked += 1
            continue

        # 连板天数过滤
        lb_days = _to_int(row.get("连板天数", row.get("涨停天数", 1)))
        if lb_days > CFG["max_board_days"]:
            reason["连板过高"] += 1
            blocked += 1
            continue

        # 换手率过滤
        turnover_val = _to_num(row.get("换手率", 0))
        if turnover_val > CFG["max_turnover"]:
            reason["换手过高"] += 1
            blocked += 1
            continue

        # 一字板/巨单封死过滤（买不进）
        fengban = _to_num(row.get("封板资金", 0))
        circ = max(_to_num(row.get("流通市值", 0)), 1)
        if circ > 0 and fengban / circ > CFG["fengban_ratio"]:
            reason["一字板"] += 1
            blocked += 1
            continue

        # 构造评分字典
        row_dict = {
            "换手率": row.get("换手率", 0),
            "成交额": row.get("成交额", 0),
            "涨跌幅": row.get("涨跌幅", 0),
            "流通市值": row.get("流通市值", 0),
            "连板天数": lb_days,
            "首次封板时间": row.get("首次封板时间", ""),
            "所属行业": row.get("所属行业", ""),
        }

        score = calc_score(row_dict, board_map)
        if score < CFG["min_score"]:
            continue

        results.append({
            "代码": code,
            "名称": name,
            "评分": score,
            "换手率": turnover_val,
            "成交额_亿": round(_to_num(row.get("成交额", 0)) / 1e8, 1),
            "涨跌幅": _to_num(row.get("涨跌幅", 0)),
            "流通市值_亿": round(_to_num(row.get("流通市值", 0)) / 1e8, 1),
            "连板天数": lb_days,
            "封板时间": str(row.get("首次封板时间", "—")),
            "行业": str(row.get("所属行业", "—")),
        })

    print(f"达标 {len(results)} 只 | 剔除{blocked} {reason}")

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
                f"{_to_num(row.get('流通市值_亿')):.0f}亿 | "
                f"{int(row.get('连板天数',1))}板 | "
                f"{row.get('封板时间','—')}"
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
