import os
import sys
import argparse
import datetime
import json
import urllib.request
import pandas as pd
import numpy as np
import akshare as ak

# ============ 配置 ============
CFG = {
    "date": None,
    "top_n": 15,
    "unlock_days": 5,       # 解禁前N天剔除
    "unlock_max": 40,       # 解禁占比>40%扣分
    "min_score": 30,        # 最低评分门槛
}

# ============ 数据获取 ============
def get_trade_date():
    if CFG["date"]:
        return CFG["date"]
    return datetime.datetime.now().strftime("%Y%m%d")

def get_limit_up_pool(date_str):
    """获取涨停池"""
    try:
        df = ak.stock_zt_pool_em(date=date_str)
        return df
    except Exception as e:
        print(f"⚠️ 获取涨停池失败: {e}")
        return pd.DataFrame()

def get_unlock_calendar():
    """获取解禁日历"""
    try:
        df = ak.stock_restricted_release_queue_em()
        return df
    except Exception as e:
