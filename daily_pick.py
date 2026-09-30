import os
import sys
import argparse
import datetime
import json
import urllib.request
import time
import pandas as pd
import numpy as np

CFG = {
    "date": None,
    "top_n": 15,
    "min_score": 75,
    "max_board_days": 3,
    "max_turnover": 28.0,
    "fengban_ratio": 0.05,
    "min_circ_mv": 15e8,
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

def is_morning_zt(v):
    """判断是否为早盘封板（10:30前）。接口返回 '093114' 或 '09:31:14'"""
    s = str(v).strip()
    if not s or s in ("—", "nan", "None", "0"):
        return False, ""
    if s.isdigit() and len(s) == 6:
        hh, mm = int(s[:2]), int(s[2:4])
        return (hh < 10) or (hh == 10 and mm <= 30), f"{s[:2]}:{s[2:4]}:{s[4:]}"
    if ":" in s:
        try:
            t = datetime.datetime.strptime(s[:8], "%H:%M:%S")
            return t.hour < 10 or (t.hour == 10 and t.minute <= 30), s[:8]
        except Exception:
            pass
    return False, s

def calc_score(d):
    """综合评分：基础分20 + 各项加权，区间[0,100]"""
    score = 20

    # 换手率：6-12最佳，>25扣分
    t = _to_num(d.get("换手率", 0))
    if 6 <= t <= 12:
        score += 18
    elif 12 < t <= 18:
        score += 12
    elif 18 < t <= 25:
        score += 4
    else:
        score -= 15

    # 成交额：>=15亿最优
    a = _to_num(d.get("成交额", 0))
    if a >= 15e8:
        score += 14
    elif a >= 8e8:
        score += 10
    elif a >= 4e8:
        score += 6
    elif a >= 1.5e8:
        score += 3
    else:
        score -= 10

    # 流通市值：30-120亿最佳弹性
    c = _to_num(d.get("流通市值", 0))
    if 30e8 <= c <= 120e8:
        score += 14
    elif 120e8 < c <= 250e8:
        score += 7
    elif c > 250e8:
        score += 2

    # 涨跌幅：20cm加分
    ch = _to_num(d.get("涨跌幅", 0))
    if ch >= 19.9:
        score += 8
    elif ch >= 9.9:
        score += 4

    # 连板：首板最优，高位扣分
    lb = _to_int(d.get("连板数", 1))
    if lb == 1:
        score += 10
    elif lb == 2:
        score += 4
    else:
        score -= 10

    # 早盘封板
    m, _ = is_morning_zt(d.get("首次封板时间", ""))
    if m:
        score += 10
    else:
        score -= 15

    # 炸板次数
    zha = _to_int(d.get("炸板次数", 0))
    if zha == 0:
        score += 8
    elif zha == 1:
        score += 3
    elif zha == 2:
        score -= 5
    else:
        score -= 15

    return max(0, min(score, 100))

def hard_filter(row):
    """硬过滤：不满足直接剔除。返回 (是否通过, 原因)"""
    name = str(row.get("名称", ""))
    if "ST" in name.upper():
        return False, "ST"
    if _to_int(row.get("连板数", 1)) > CFG["max_board_days"]:
        return False, f"连板>3"
    if _to_num(row.get("换手率", 0)) > CFG["max_turnover"]:
        return False, "换手过高"
    circ = max(_to_num(row.get("流通市值", 0)), 1)
    if circ > 0 and _to_num(row.get("封板资金", 0)) / circ > CFG["fengban_ratio"]:
        return False, "一字板"
    if _to_num(row.get("流通市值", 0)) < CFG["min_circ_mv"]:
        return False, "市值过小"
    # 尾盘板+多次炸板：剔除
    m, _ = is_morning_zt(row.get("首次封板时间", ""))
    if not m and _to_int(row.get("炸板次数", 0)) >= 2:
        return False, "尾盘偷袭"
    return True, ""

def get_limit_up_pool(date_str):
    for attempt in range(3):
        try:
            import akshare as ak
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
            import akshare as ak
            df = ak.stock_restricted_release_queue_em()
            if df is not None and not df.empty:
                code_col = next((c for c in df.columns if "代码" in c or "code" in c.lower()), df.columns[0])
                return set(str(x).zfill(6) for x in df[code_col].tolist())
        except Exception as e:
            print(f"⚠️ 解禁接口失败（尝试 {attempt+1}/2）: {e}")
        if attempt < 1:
            time.sleep(3)
    return set()

def pick():
    date_str = get_trade_date()
    print(f"\n== 选股日期：{date_str} | 模式：first ==")

    zt_df = get_limit_up_pool(date_str)
    if zt_df is None or zt_df.empty:
        print("📭 涨停池为空（非交易日/接口持续失败）")
        return pd.DataFrame()
    print(f"涨停池 {len(zt_df)} 只")

    unlock_set = load_unlock_set()
    if unlock_set:
        print(f"🛡️ 解禁池 {len(unlock_set)} 只")
    else:
        print("ℹ️ 解禁数据未取到，跳过解禁风控")

    cols = zt_df.columns.tolist()
    print(f"可用字段: {cols}")

    results, blocked = [], 0
    reason = {"ST": 0, "连板过高": 0, "换手过高": 0, "一字板": 0,
              "市值过小": 0, "尾盘偷袭": 0, "解禁": 0}

    for _, row in zt_df.iterrows():
        code = str(row.get("代码", "")).zfill(6)
        name = str(row.get("名称", ""))

        if code in unlock_set:
            reason["解禁"] += 1
            blocked += 1
            continue

        ok, r = hard_filter(row)
        if not ok:
            reason[r] = reason.get(r, 0) + 1
            blocked += 1
            continue

        d = {
            "换手率": row.get("换手率", 0),
            "成交额": row.get("成交额", 0),
            "涨跌幅": row.get("涨跌幅", 0),
            "流通市值": row.get("流通市值", 0),
            "连板数": _to_int(row.get("连板数", 1)),
            "首次封板时间": row.get("首次封板时间", ""),
            "炸板次数": row.get("炸板次数", 0),
        }
        score = calc_score(d)
        if score < CFG["min_score"]:
            continue

        m, td = is_morning_zt(row.get("首次封板时间", ""))
        results.append({
            "代码": code,
            "名称": name,
            "评分": score,
            "换手率": round(_to_num(row.get("换手率", 0)), 1),
            "成交额_亿": round(_to_num(row.get("成交额", 0)) / 1e8, 1),
            "涨跌幅": round(_to_num(row.get("涨跌幅", 0)), 2),
            "流通市值_亿": round(_to_num(row.get("流通市值", 0)) / 1e8, 1),
            "连板天数": _to_int(row.get("连板数", 1)),
            "封板时间": td,
            "炸板次数": _to_int(row.get("炸板次数", 0)),
            "早盘": "🌅" if m else "🌙",
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
                f"{row.get('换手率', 0):.1f}%换 | {row.get('成交额_亿', 0):.1f}亿 | "
                f"{int(row.get('连板天数', 1))}板 | "
                f"{row.get('封板时间', '—')} {row.get('早盘', '')}"
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
