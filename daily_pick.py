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
    "min_score": 65,              # 再提一点
    "max_board_days": 3,
    "max_turnover": 28.0,
    "fengban_ratio": 0.05,
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

def is_morning_zt(fengban_time_val):
    """
    判断是否为早盘封板。
    接口返回格式可能是 093114（int）或 '09:31:14'（str）
    """
    s = str(fengban_time_val).strip()
    if not s or s in ("—", "nan", "None", "0"):
        return False, ""
    # 6位数字格式：093114
    if s.isdigit() and len(s) == 6:
        hh = int(s[:2])
        mm = int(s[2:4])
        display = f"{s[:2]}:{s[2:4]}:{s[4:]}"
        return (hh < 10) or (hh == 10 and mm <= 30), display
    # 字符串格式：09:31:14
    if ":" in s:
        try:
            t = datetime.datetime.strptime(s[:8], "%H:%M:%S")
            display = s[:8]
            return t.hour < 10 or (t.hour == 10 and t.minute <= 30), display
        except Exception:
            pass
    return False, s

def calc_score(row_dict):
    score = 35  # 基础分略降

    # 换手率
    turnover = _to_num(row_dict.get("换手率", 0))
    if 5 <= turnover <= 12:
        score += 22
    elif 12 < turnover <= 18:
        score += 14
    elif 18 < turnover <= CFG["max_turnover"]:
        score += 6
    elif turnover > CFG["max_turnover"]:
        score -= 15

    # 成交额
    amount = _to_num(row_dict.get("成交额", 0))
    if amount >= 15e8:
        score += 16
    elif amount >= 8e8:
        score += 12
    elif amount >= 3e8:
        score += 6

    # 流通市值
    circ_mv = _to_num(row_dict.get("流通市值", 0))
    if 30e8 <= circ_mv <= 150e8:
        score += 16
    elif 150e8 < circ_mv <= 300e8:
        score += 8
    elif circ_mv > 300e8:
        score += 3

    # 涨跌幅（20cm加分）
    change = _to_num(row_dict.get("涨跌幅", 0))
    if change >= 19.9:
        score += 12
    elif change >= 9.9:
        score += 6

    # 连板天数（首板加分，高位减分）
    lb_days = _to_int(row_dict.get("连板数", 1))
    if lb_days == 1:
        score += 12
    elif lb_days == 2:
        score += 6
    elif lb_days == 3:
        score += 2
    # 4板以上已经在硬过滤踢掉了

    # 早盘封板
    is_morning, _ = is_morning_zt(row_dict.get("首次封板时间", ""))
    if is_morning:
        score += 12

    # 炸板次数（越少越好）
    zha_count = _to_int(row_dict.get("炸板次数", 0))
    if zha_count == 0:
        score += 8
    elif zha_count == 1:
        score += 3
    elif zha_count >= 3:
        score -= 10

    return max(0, min(score, 100))

def pick():
    date_str = get_trade_date()
    print(f"\n== 选股日期：{date_str} | 模式：first ==")

    zt_df = get_limit_up_pool(date_str)
    if zt_df is None or zt_df.empty:
        print("📭 涨停池为空")
        return pd.DataFrame()
    print(f"涨停池 {len(zt_df)} 只")

    unlock_set = load_unlock_set()
    if unlock_set:
        print(f"🛡️ 解禁池 {len(unlock_set)} 只")
    else:
        print("ℹ️ 解禁数据未取到，跳过解禁风控")

    cols = zt_df.columns.tolist()
    print(f"可用字段: {cols}")

    results = []
    blocked = 0
    reason = {"连板过高": 0, "换手过高": 0, "一字板": 0, "ST": 0, "解禁": 0}

    for _, row in zt_df.iterrows():
        code = str(row.get("代码", "")).zfill(6)
        name = str(row.get("名称", ""))

        # ST 剔除
        if "ST" in name.upper():
            reason["ST"] += 1
            blocked += 1
            continue

        # 解禁剔除
        if code in unlock_set:
            reason["解禁"] += 1
            blocked += 1
            continue

        # 连板天数（字段名：连板数）
        lb_days = _to_int(row.get("连板数", 1))
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

        # 一字板/巨单封死过滤
        fengban = _to_num(row.get("封板资金", 0))
        circ = max(_to_num(row.get("流通市值", 0)), 1)
        if circ > 0 and fengban / circ > CFG["fengban_ratio"]:
            reason["一字板"] += 1
            blocked += 1
            continue

        # 构造评分字典（字段名全部对齐）
        row_dict = {
            "换手率": row.get("换手率", 0),
            "成交额": row.get("成交额", 0),
            "涨跌幅": row.get("涨跌幅", 0),
            "流通市值": row.get("流通市值", 0),
            "连板数": lb_days,
            "首次封板时间": row.get("首次封板时间", ""),
            "炸板次数": row.get("炸板次数", 0),
            "所属行业": row.get("所属行业", ""),
        }

        score = calc_score(row_dict)
        if score < CFG["min_score"]:
            continue

        is_morning, time_display = is_morning_zt(row.get("首次封板时间", ""))

        results.append({
            "代码": code,
            "名称": name,
            "评分": score,
            "换手率": round(turnover_val, 1),
            "成交额_亿": round(_to_num(row.get("成交额", 0)) / 1e8, 1),
            "涨跌幅": round(_to_num(row.get("涨跌幅", 0)), 2),
            "流通市值_亿": round(_to_num(row.get("流通市值", 0)) / 1e8, 1),
            "连板天数": lb_days,
            "封板时间": time_display,
            "炸板次数": _to_int(row.get("炸板次数", 0)),
            "早盘": "🌅" if is_morning else "🌙",
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
                f"{row.get('换手率',0):.1f}%换 | "
                f"{row.get('成交额_亿',0):.1f}亿 | "
                f"{int(row.get('连板天数',1))}板 | "
                f"{row.get('封板时间','—')} {row.get('早盘','')}"
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
