#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每日选股推送 - 主线增强版
功能：盘后涨停池筛选 → 板块主线聚类 → 主力资金验证 → 形态过滤 → 企微推送
支持模式：fast(无K线) / full(含均线) / no_fund(无资金接口)
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

# ============ 配置常量 ============
MIN_SCORE = 75                    # 最低评分阈值
DEFAULT_TOP_N = 15               # 默认推送数量
MAX_BOARDS = 3                   # 最大连板数过滤
MAX_TURNOVER = 28.0              # 最大换手率(%)
MIN_MARKET_CAP = 15.0            # 最小市值(亿)
MAIN_BOARD_MIN_STOCKS = 3        # 主线板块最少涨停家数
FUND_FLOW_DAYS = 5               # 资金流向观察天数
REQUEST_TIMEOUT = 15             # 单次请求超时(秒)
RETRY_COUNT = 2                  # 接口重试次数
RETRY_DELAY = 3                  # 重试间隔(秒)

# 硬过滤条件
HARD_FILTERS = {
    "st": True,           # 剔除ST
    "max_boards": 3,      # 连板>3剔除
    "max_turnover": 28.0, # 换手>28%剔除
    "one_word": True,     # 剔除一字板
    "min_market_cap": 15.0,# 市值<15亿剔除
    "late_afternoon": True,# 剔除尾盘偷袭(14:30后首次封板)
}

# 评分权重
SCORE_WEIGHTS = {
    "board_count": 5,        # 连板数加分(首板0，2板10...)
    "sector_main": 20,       # 主线板块加分
    "fund_flow_5d": 15,      # 5日资金净流入
    "fund_flow_10d": 10,     # 10日资金净流入
    "morning_lobby": 10,     # 早盘封板(10:00前)
    "no_open": 10,           # 未开板(炸板次数=0)
    "ma_bull": 15,           # 均线多头
    "return_shot": 15,       # 回马枪形态
    "market_cap_bonus": 5,   # 市值适中(30-100亿)
    "turnover_good": 5,      # 换手健康(5-15%)
}


def log(msg: str, level: str = "INFO"):
    """带时间戳的日志"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")


def retry_on_failure(func, *args, **kwargs):
    """带重试的接口调用"""
    last_error = None
    for i in range(RETRY_COUNT):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_error = e
            if i < RETRY_COUNT - 1:
                log(f"⚠ 请求失败({i+1}/{RETRY_COUNT}): {e}, {RETRY_DELAY}秒后重试...", "WARN")
                time.sleep(RETRY_DELAY)
    raise last_error


def parse_ftime(ftime_str) -> Optional[datetime]:
    """解析首次封板时间(6位数字如093114 → HH:MM:SS)"""
    if pd.isna(ftime_str) or ftime_str is None:
        return None
    try:
        s = str(int(float(ftime_str))).zfill(6)
        if len(s) >= 4:
            hour = int(s[:2])
            minute = int(s[2:4])
            return datetime(2020, 1, 1, hour, minute)
    except (ValueError, TypeError):
        pass
    return None


def is_one_word_board(row: pd.Series) -> bool:
    """判断是否一字板：涨停价=开盘价 且 未开板(炸板次数=0)"""
    try:
        if row.get("炸板次数", 1) != 0:
            return False
        # 涨停价 ≈ 昨收 * 1.1 (A股10%)，这里简化判断
        # 实际可用：开盘价/昨收 ≥ 1.098 视为一字板
        if "开盘价" in row and "昨收价" in row and row["昨收价"] > 0:
            return (row["开盘价"] / row["昨收价"]) >= 1.098
    except Exception:
        pass
    return False


def is_late_afternoon(ftime_str) -> bool:
    """是否尾盘偷袭(14:30后首次封板)"""
    dt = parse_ftime(ftime_str)
    if dt is None:
        return False
    cutoff = datetime(dt.year, dt.month, dt.day, 14, 30)
    return dt > cutoff


def get_limit_up_pool(trade_date: str = None) -> pd.DataFrame:
    """获取涨停池数据"""
    try:
        import akshare as ak
        log("📡 获取涨停池数据...")
        
        if trade_date:
            # 指定日期
            df = retry_on_failure(ak.stock_zt_pool_em, date=trade_date)
        else:
            # 当日
            df = retry_on_failure(ak.stock_zt_pool_em)
        
        if df is None or df.empty:
            log("❌ 涨停池为空", "ERROR")
            return pd.DataFrame()
        
        log(f"✅ 涨停池 {len(df)} 只")
        
        # 字段名标准化（根据实际akshare返回字段调整）
        # 常见字段：代码,名称,涨跌幅,最新价,成交额,流通市值,总市值,换手率,封板资金,首次封板时间,最后封板时间,炸板次数,连板数,所属行业
        field_mapping = {
            "代码": "code",
            "名称": "name", 
            "涨跌幅": "pct_change",
            "最新价": "price",
            "成交额": "amount",
            "流通市值": "circ_market_cap",
            "总市值": "total_market_cap",
            "换手率": "turnover",
            "封板资金": "seal_amount",
            "首次封板时间": "first_time",
            "最后封板时间": "last_time",
            "炸板次数": "open_times",
            "连板数": "board_count",
            "所属行业": "industry",
        }
        
        # 重命名存在的字段
        for old, new in field_mapping.items():
            if old in df.columns:
                df = df.rename(columns={old: new})
        
        # 确保必要字段存在
        required = ["code", "name"]
        for r in required:
            if r not in df.columns:
                log(f"❌ 缺少必要字段: {r}", "ERROR")
                return pd.DataFrame()
        
        # 补充缺失字段默认值
        if "board_count" not in df.columns:
            df["board_count"] = 1
        if "turnover" not in df.columns:
            df["turnover"] = 0.0
        if "total_market_cap" not in df.columns:
            df["total_market_cap"] = 0.0
        if "first_time" not in df.columns:
            df["first_time"] = None
        if "open_times" not in df.columns:
            df["open_times"] = 0
        if "industry" not in df.columns:
            df["industry"] = "未知"
            
        return df
        
    except Exception as e:
        log(f"❌ 获取涨停池失败: {e}", "ERROR")
        traceback.print_exc()
        return pd.DataFrame()


def get_sector_data() -> Dict[str, int]:
    """获取板块涨停分布，返回 {板块名: 涨停家数}"""
    try:
        import akshare as ak
        log("📡 获取板块数据...")
        
        df = retry_on_failure(ak.stock_board_industry_name_em)
        if df is None or df.empty:
            log("⚠ 板块接口返回空，跳过板块加分", "WARN")
            return {}
        
        # 统计每个板块的涨停家数
        sector_counts = {}
        # 这里需要遍历涨停池中的每只股票，获取其所属板块
        # 由于akshare没有直接返回板块涨停数，我们需要从涨停池的industry字段统计
        # 或者调用 stock_board_industry_cons_em 逐个板块查
        # 简化：从涨停池的industry字段统计
        return sector_counts  # 实际统计在main中做
        
    except Exception as e:
        log(f"⚠ 板块接口失败: {e}", "WARN")
        return {}


def get_fund_flow(code: str, days: int = 5) -> Tuple[float, float]:
    """获取主力资金净流入(5日, 10日)，返回(5日净额, 10日净额) 单位:万"""
    try:
        import akshare as ak
        
        # 尝试获取个股资金流向
        df = retry_on_failure(ak.stock_individual_fund_flow, stock=code)
        if df is None or df.empty:
            return 0.0, 0.0
        
        # 假设返回字段包含日期和主力净流入
        if "主力净流入" in df.columns:
            df = df.sort_values("日期", ascending=False)
            net_5d = df["主力净流入"].head(days).sum()
            net_10d = df["主力净流入"].head(10).sum()
            return float(net_5d), float(net_10d)
        
        return 0.0, 0.0
        
    except Exception as e:
        log(f"⚠ {code} 资金流向获取失败: {e}", "WARN")
        return 0.0, 0.0


def get_kline_data(code: str, days: int = 30) -> Optional[pd.DataFrame]:
    """获取K线数据用于均线计算"""
    try:
        import akshare as ak
        
        df = retry_on_failure(ak.stock_zh_a_hist, symbol=code, period="daily", 
                              start_date=(datetime.now() - timedelta(days=days*2)).strftime("%Y%m%d"),
                              end_date=datetime.now().strftime("%Y%m%d"))
        if df is None or df.empty:
            return None
        
        # 字段标准化
        if "收盘" in df.columns:
            df = df.rename(columns={"收盘": "close"})
        if "日期" in df.columns:
            df["日期"] = pd.to_datetime(df["日期"])
            df = df.sort_values("日期")
            
        return df.tail(days)
        
    except Exception as e:
        log(f"⚠ {code} K线获取失败: {e}", "WARN")
        return None


def check_ma_bull(kline_df: pd.DataFrame) -> bool:
    """检查均线多头排列(5>10>20)"""
    if kline_df is None or len(kline_df) < 20:
        return False
    
    try:
        close = kline_df["close"]
        ma5 = close.rolling(5).mean().iloc[-1]
        ma10 = close.rolling(10).mean().iloc[-1]
        ma20 = close.rolling(20).mean().iloc[-1]
        
        return ma5 > ma10 > ma20
    except Exception:
        return False


def check_return_shot(kline_df: pd.DataFrame, current_price: float) -> bool:
    """检查回马枪形态：前期涨停后回调，今日再次涨停"""
    if kline_df is None or len(kline_df) < 10:
        return False
    
    try:
        # 简化判断：10日内有过涨停(涨幅≥9.8%)，之后回调，今日再次涨停
        close = kline_df["close"]
        pct_changes = close.pct_change() * 100
        
        # 找10日内涨停
        recent_limit_up = (pct_changes >= 9.8).any()
        if not recent_limit_up:
            return False
        
        # 回调幅度：从最高到最低回调至少5%
        recent_high = close.tail(10).max()
        recent_low = close.tail(10).min()
        if recent_high > 0:
            pullback = (recent_high - recent_low) / recent_high
            return pullback >= 0.05
        
        return False
    except Exception:
        return False


def hard_filter(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """硬过滤，返回过滤后的df和被剔除原因"""
    if df.empty:
        return df, []
    
    original_len = len(df)
    reasons = []
    
    # 1. 剔除ST
    if HARD_FILTERS["st"]:
        mask = df["name"].str.contains("ST|st|退", case=False, na=False)
        if mask.any():
            reasons.append(f"ST: {mask.sum()}只")
            df = df[~mask]
    
    # 2. 连板>3
    if "board_count" in df.columns:
        mask = df["board_count"] > HARD_FILTERS["max_boards"]
        if mask.any():
            reasons.append(f"连板>3: {mask.sum()}只")
            df = df[~mask]
    
    # 3. 换手率>28%
    if "turnover" in df.columns:
        mask = df["turnover"] > HARD_FILTERS["max_turnover"]
        if mask.any():
            reasons.append(f"换手>28%: {mask.sum()}只")
            df = df[~mask]
    
    # 4. 一字板
    if HARD_FILTERS["one_word"]:
        mask = df.apply(is_one_word_board, axis=1)
        if mask.any():
            reasons.append(f"一字板: {mask.sum()}只")
            df = df[~mask]
    
    # 5. 市值<15亿
    if "total_market_cap" in df.columns:
        mask = df["total_market_cap"] < HARD_FILTERS["min_market_cap"]
        if mask.any():
            reasons.append(f"市值<15亿: {mask.sum()}只")
            df = df[~mask]
    
    # 6. 尾盘偷袭(14:30后首次封板)
    if HARD_FILTERS["late_afternoon"]:
        mask = df["first_time"].apply(lambda x: is_late_afternoon(x))
        if mask.any():
            reasons.append(f"尾盘偷袭: {mask.sum()}只")
            df = df[~mask]
    
    filtered_count = original_len - len(df)
    log(f"🔍 硬过滤: 剔除{filtered_count}只 ({', '.join(reasons)})")
    
    return df, reasons


def score_stock(row: pd.Series, sector_counts: Dict[str, int], 
                use_fund: bool = True, use_ma: bool = False) -> Tuple[int, List[str]]:
    """给单只股票打分，返回(总分, 加分原因列表)"""
    score = 0
    reasons = []
    
    # 1. 连板加分
    board_count = row.get("board_count", 1)
    if board_count >= 2:
        board_score = min(board_count * SCORE_WEIGHTS["board_count"], 15)
        score += board_score
        reasons.append(f"连板{board_count}层(+{board_score})")
    
    # 2. 主线板块加分
    industry = row.get("industry", "未知")
    sector_up_count = sector_counts.get(industry, 0)
    if sector_up_count >= MAIN_BOARD_MIN_STOCKS:
        score += SCORE_WEIGHTS["sector_main"]
        reasons.append(f"主线板块({industry}{sector_up_count}家)(+{SCORE_WEIGHTS['sector_main']})")
    
    # 3. 资金流向加分
    if use_fund:
        fund_5d, fund_10d = get_fund_flow(row["code"])
        if fund_5d > 0:
            fund_score = min(int(fund_5d / 1000), SCORE_WEIGHTS["fund_flow_5d"])  # 每千万1分
            score += fund_score
            reasons.append(f"5日资金+{fund_5d:.0f}万(+{fund_score})")
        if fund_10d > 0:
            fund_score_10 = min(int(fund_10d / 2000), SCORE_WEIGHTS["fund_flow_10d"])
            score += fund_score_10
            reasons.append(f"10日资金+{fund_10d:.0f}万(+{fund_score_10})")
    
    # 4. 早盘封板(10:00前)
    ftime = parse_ftime(row.get("first_time"))
    if ftime and ftime.hour < 10:
        score += SCORE_WEIGHTS["morning_lobby"]
        reasons.append(f"早盘封板({ftime.strftime('%H:%M')})(+{SCORE_WEIGHTS['morning_lobby']})")
    
    # 5. 未开板(炸板次数=0)
    if row.get("open_times", 1) == 0:
        score += SCORE_WEIGHTS["no_open"]
        reasons.append("未开板(+10)")
    
    # 6. 均线多头
    if use_ma:
        kline = get_kline_data(row["code"])
        if check_ma_bull(kline):
            score += SCORE_WEIGHTS["ma_bull"]
            reasons.append("均线多头(+15)")
        
        # 7. 回马枪形态
        if check_return_shot(kline, row.get("price", 0)):
            score += SCORE_WEIGHTS["return_shot"]
            reasons.append("回马枪(+15)")
    
    # 8. 市值适中(30-100亿)
    mcap = row.get("total_market_cap", 0)
    if 30 <= mcap <= 100:
        score += SCORE_WEIGHTS["market_cap_bonus"]
        reasons.append(f"市值适中({mcap:.0f}亿)(+5)")
    
    # 9. 换手健康(5-15%)
    turnover = row.get("turnover", 0)
    if 5 <= turnover <= 15:
        score += SCORE_WEIGHTS["turnover_good"]
        reasons.append(f"换手健康({turnover:.1f}%)(+5)")
    
    return min(score, 100), reasons  # 封顶100


def send_wecom_webhook(webhook_url: str, content: str, msg_type: str = "markdown"):
    """发送企微群机器人消息"""
    if not webhook_url:
        log("⚠ 未配置 WECOM_WEBHOOK，跳过推送", "WARN")
        return False
    
    try:
        import urllib.request
        import json
        
        payload = {
            "msgtype": msg_type,
            msg_type: {
                "content": content
            }
        }
        
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(webhook_url, data=data, headers={"Content-Type": "application/json"})
        
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            if result.get("errcode") == 0:
                log("✅ 企微推送成功")
                return True
            else:
                log(f"❌ 企微推送失败: {result}", "ERROR")
                return False
                
    except Exception as e:
        log(f"❌ 企微推送异常: {e}", "ERROR")
        return False


def format_message(df: pd.DataFrame, mode: str, tag: str, filtered_reasons: List[str]) -> str:
    """格式化推送消息"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    msg = f"""## 📈 每日选股推送 - {tag}
**时间**: {now}
**模式**: {mode}
**筛选结果**: {len(df)} 只 (评分≥{MIN_SCORE})

### 🏆 候选标的
"""
    
    if df.empty:
        msg += "\n> 今日无符合标准的标的，建议空仓观望。\n"
    else:
        for i, (_, row) in enumerate(df.iterrows(), 1):
            if i > 15:  # 最多展示15只
                break
            
            msg += f"\n**{i}. {row['name']}** (`{row['code']}`)\n"
            msg += f"- 评分: **{row['score']}** | 板块: {row.get('industry', 'N/A')}\n"
            msg += f"- 连板: {row.get('board_count', 1)} | 换手: {row.get('turnover', 0):.1f}%\n"
            msg += f"- 市值: {row.get('total_market_cap', 0):.0f}亿\n"
            
            # 加分原因
            if "reasons" in row and row["reasons"]:
                reasons_str = "、".join(row["reasons"][:3])  # 最多3个原因
                msg += f"- 亮点: {reasons_str}\n"
    
    msg += f"\n### 📊 过滤统计\n"
    if filtered_reasons:
        for reason in filtered_reasons:
            msg += f"- {reason}\n"
    
    msg += f"\n> 💡 以上为初筛结果，不构成投资建议。详细数据见仓库 results/ 目录。"
    
    return msg


def save_results(df: pd.DataFrame, mode: str, tag: str):
    """保存结果到文件"""
    output_dir = os.environ.get("OUTPUT_DIR", "results")
    os.makedirs(output_dir, exist_ok=True)
    
    date_str = datetime.now().strftime("%Y%m%d")
    filename = f"{output_dir}/pick_{date_str}_{mode}.csv"
    
    if not df.empty:
        # 保存关键字段
        save_cols = ["code", "name", "score", "industry", "board_count", "turnover", "total_market_cap"]
        save_cols = [c for c in save_cols if c in df.columns]
        df[save_cols].to_csv(filename, index=False, encoding="utf-8-sig")
        log(f"💾 结果已保存: {filename}")
    else:
        log("📭 无结果可保存")


def main():
    parser = argparse.ArgumentParser(description="每日选股推送")
    parser.add_argument("--no-ma", action="store_true", help="跳过均线多头和回马枪检查(加速)")
    parser.add_argument("--no-fund", action="store_true", help="跳过资金流向验证")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP_N, help="推送前N只")
    parser.add_argument("--date", type=str, default="", help="指定日期(YYYYMMDD)")
    parser.add_argument("--min-score", type=int, default=MIN_SCORE, help="最低评分阈值")
    args = parser.parse_args()
    
    mode = "full"
    if args.no_ma and args.no_fund:
        mode = "fast"
    elif args.no_fund:
        mode = "no_fund"
    elif args.no_ma:
        mode = "no_ma"
    
    tag = os.environ.get("GITHUB_RUN_ID", "local")
    if tag != "local":
        tag = f"run-{tag}"
    
    log(f"🚀 启动选股 | 模式={mode} | 日期={args.date or '今日'}")
    
    try:
        # 1. 获取涨停池
        df = get_limit_up_pool(args.date)
        if df.empty:
            log("❌ 无涨停数据，退出", "ERROR")
            sys.exit(1)
        
        # 2. 板块统计
        sector_counts = {}
        if "industry" in df.columns:
            sector_counts = df["industry"].value_counts().to_dict()
            log(f"📊 板块分布: {len(sector_counts)}个板块")
        
        # 3. 硬过滤
        df, filtered_reasons = hard_filter(df)
        if df.empty:
            log("📭 过滤后无标的")
            msg = format_message(df, mode, tag, filtered_reasons)
            webhook = os.environ.get("WECOM_WEBHOOK", "")
            send_wecom_webhook(webhook, msg)
            sys.exit(0)
        
        # 4. 打分
        log("📝 开始评分...")
        scores = []
        all_reasons = []
        
        for _, row in df.iterrows():
            score, reasons = score_stock(row, sector_counts, 
                                        use_fund=not args.no_fund, 
                                        use_ma=not args.no_ma)
            scores.append(score)
            all_reasons.append(reasons)
        
        df["score"] = scores
        df["reasons"] = all_reasons
        
        # 5. 过滤低分
        df_filtered = df[df["score"] >= args.min_score].copy()
        df_filtered = df_filtered.sort_values("score", ascending=False)
        
        log(f"📊 评分完成: {len(df_filtered)} 只 ≥ {args.min_score}分")
        
        # 6. 取前N只
        top_df = df_filtered.head(args.top)
        
        # 7. 保存结果
        save_results(top_df, mode, tag)
        
        # 8. 推送
        msg = format_message(top_df, mode, tag, filtered_reasons)
        webhook = os.environ.get("WECOM_WEBHOOK", "")
        send_wecom_webhook(webhook, msg)
        
        log("✅ 选股完成")
        
    except Exception as e:
        log(f"❌ 运行异常: {e}", "ERROR")
        traceback.print_exc()
        
        # 发送错误通知
        error_msg = f"## ❌ 选股脚本异常\n**时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n**错误**: {str(e)[:200]}"
        webhook = os.environ.get("WECOM_WEBHOOK", "")
        send_wecom_webhook(webhook, error_msg)
        
        sys.exit(1)


if __name__ == "__main__":
    main()
