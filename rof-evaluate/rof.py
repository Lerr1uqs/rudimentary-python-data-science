import os 
import tushare as ts
import pandas as pd
import numpy as np
import sys
from typing import cast

TOKEN_PATH = os.path.expanduser('~/.tushare.token')
with open(TOKEN_PATH, 'r') as f:
    token = f.read().strip()
    ts.set_token(token)
    pro = ts.pro_api()

START = "20180401"
END   = "20240430"
TICKER = "002367.SZ"
if len(sys.argv) > 1:
    TICKER = sys.argv[1]

name_df: pd.DataFrame = pro.stock_basic(ts_code=TICKER, exchange='', list_status='L', fields='name')
print(name_df["name"].iloc[0])
'''
青岛啤酒
               end_date       rof   returns
3   2020-12-31 00:00:00  0.130363  0.016789
4   2021-03-31 00:00:00  0.157291  0.024405
5   2021-06-30 00:00:00  0.158806  0.018142
6   2021-09-30 00:00:00  0.163529  0.027516
7   2021-12-31 00:00:00  0.179243  0.024483
8   2022-03-31 00:00:00  0.182213  0.032366
9   2022-06-30 00:00:00  0.197602  0.026675
10  2022-09-30 00:00:00  0.210000  0.027992
11  2022-12-31 00:00:00  0.202665  0.026754
12  2023-03-31 00:00:00  0.220886  0.026032
13  2023-06-30 00:00:00  0.222884  0.032149
14  2023-09-30 00:00:00  0.222394  0.039573
15  2023-12-31 00:00:00  0.217259  0.045712
16  2024-03-31 00:00:00  0.216942  0.042346
'''
# TICKER = "000333.SZ" # 美的集团

factor: pd.DataFrame = pro.daily_basic(
    ts_code=TICKER, 
    start_date=START, 
    end_data=END, 
    fields='trade_date,pe,pb,total_mv'
)
factor["total_mv"] = factor["total_mv"] * 1e4 # 这里的单位是万 要转为普通单位
current_market_value = factor["total_mv"].iloc[0]
# data: pd.DataFrame = ts.pro_bar(ts_code=TICKER, adj='qfq', start_date=START, end_date=END)

fina_col = ["end_date", "roe", "roa"]
fina_ind = pro.fina_indicator(
    ts_code=TICKER, 
    start_date=START, 
    end_date=END
)[fina_col]

fina_ind = fina_ind.drop_duplicates(subset='end_date') # 有多个end_date
# fina_ind["end_date"] = pd.to_datetime(fina_ind["end_date"])
# fina_ind.sort_values("end_date", inplace=True, ascending=True)
# fina_ind.reset_index(drop=True, inplace=True)
# print(fina_ind)

# 利润表
income: pd.DataFrame = pro.income(
    ts_code=TICKER, 
    start_date=START, 
    end_date=END, 
    fields=",".join([
        "end_date",
        # "net_after_nr_lp_correct", # 扣非净利润
        "n_income_attr_p", # 净利润(不含少数股东损益)
    ])
)

# 资产负债表
balancesheet: pd.DataFrame = pro.balancesheet(
    ts_code=TICKER, 
    start_date=START, 
    end_date=END, 
    fields=",".join([
        "end_date",
        "total_cur_assets", # 流动资产合计
        "total_nca",        # 非流动资产合计 Total Non-Current Assets
        "total_liab",       # 负债合计
        "inventories",      # 存货
    ])
)

fina_table = pd.merge(income, balancesheet, on='end_date')
fina_table = pd.merge(fina_table, fina_ind, on='end_date')
fina_table.drop_duplicates(subset='end_date', inplace=True)
# fina_table = fina_table[fina_table["end_date"].str.endswith("1231")] # 只估算每年最后一年 如果要滚动 就要加其他逻辑 因为净收入是每季度叠加的
fina_table["end_date"] = pd.to_datetime(fina_table["end_date"])
fina_table.sort_values("end_date", inplace=True)
fina_table.reset_index(drop=True, inplace=True)

fina_table["ca/liab"] = fina_table["total_cur_assets"] / fina_table["total_liab"] # current assets / liabilities
# current assets without inventories(房地产存货很多)
fina_table["cawi/liab"] = (fina_table["total_cur_assets"] - fina_table["inventories"]) / fina_table["total_liab"] 

fina_table["n_income_attr_p_diff"] = 0.0

# NOTE: 使用rolling sum的时候要确保 最开始有四个完整的季度
fina_table = fina_table[fina_table["end_date"].dt.year > fina_table["end_date"].iloc[0].year]
fina_table.reset_index(drop=True, inplace=True)
# 计算ttm
for (idx, row) in fina_table.iterrows():
    if idx == 0:
        # NOTE: 这里是start为一季度开始的情况才可以这样
        fina_table.at[0, "n_income_attr_p_diff"] = row["n_income_attr_p"]
        # fina_ind.at[0, "equity_diff"] = row["total_hldr_eqy_exc_min_int"]
        continue

    idx = cast(int, idx)
    # 如果上一个row和当前处于同一年 进行diff

    if fina_table.at[idx-1, "end_date"].year == row["end_date"].year:
        fina_table.at[idx, "n_income_attr_p_diff"] = row["n_income_attr_p"] - fina_table.at[idx-1, "n_income_attr_p"]
    else:
        # 否则 直接赋值
        fina_table.at[idx, "n_income_attr_p_diff"] = row["n_income_attr_p"]

fina_table["n_income_attr_p_ttm"] = fina_table["n_income_attr_p_diff"].rolling(window=4).sum()
# import pdb; pdb.set_trace()
# fina_table["n_income_attr_p_diff"] = fina_table["n_income_attr_p"].diff()
# fina_table["n_income_attr_p_diff_ttm"] = fina_table["n_income_attr_p_diff"].rolling(window=4).sum()

factor.rename(columns={"trade_date": "end_date"}, inplace=True)
factor.loc[:, "end_date"] = pd.to_datetime(factor["end_date"], errors='coerce')
factor.sort_values("end_date", inplace=True)
factor.reset_index(drop=True, inplace=True)


# fina_table = pd.merge(factor, fina_table, on='end_date', how="inner")
fina_table["total_mv"] = 0.0
for (idx, row) in fina_table.iterrows():
    mv_list = factor[factor["end_date"] <= row["end_date"]]["total_mv"]
    if len(mv_list) == 0:
        fina_table.at[idx, "total_mv"] = np.NaN
    else:
        fina_table.at[idx, "total_mv"] = mv_list.iloc[-1]

fina_table.dropna(inplace=True)
# print(fina_table)

# return on fixed assets
fina_table["rof"] = fina_table["n_income_attr_p_ttm"] / fina_table["total_nca"] # 净利润 / 非流动性资产
# rof = pft / nca
# pb = (mv - (ca - liab)) / nca
# (mv - (ca - liab)) / nca = rof / ret
# ret = rof * nca / (mv - (ca - liab))
# ret = pft / (mv - (ca - liab))
# 负债越高 回报率越低


# (市值 - (流动资产 - 负债)) / 非流动性资产 = rof / 收益率
# 流动资产就是 现金 + 应收
# 收益率 = rof * 非流动性资产 / 纯资产市值
# 这里的假设是 流动资产要能够偿还负债 可以用ca/liab来衡量
# assets_market_value : 去除市值中对负债和现金的估计
fina_table["assets_market_value"] = fina_table["total_mv"] - (fina_table["total_cur_assets"] - fina_table["total_liab"])
if (fina_table["assets_market_value"] < 0).any():
    print("[-] Insolvency")
    import sys
    sys.exit(1)

fina_table["returns"] = fina_table["rof"] * fina_table["total_nca"] / fina_table["assets_market_value"]
# print(fina_table[["end_date", "inventories", "total_cur_assets", "total_nca", "total_liab", "cawi/liab", "total_mv", "rof", "assets_market_value"]])
# print(fina_table)
from typing import List
def convert_to_decimal(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:

    cloned = df.copy()
    
    def cvt(s: pd.Series) -> pd.Series:
        r = []
        for i in s:
            r.append(f"{i:.2%}")
        return pd.Series(r)

    for c in cols:
        cloned[c] = cvt(cloned[c])

    return cloned

# 统一调整为小数格式
fina_table["roe"] = fina_table["roe"] / 100
fina_table["roa"] = fina_table["roa"] / 100
fina_table.reset_index(drop=True, inplace=True)

fina_table_cloned = convert_to_decimal(fina_table, ["roe", "roa", "rof", "returns"])
# print(fina_table[["end_date", "ca/liab", "cawi/liab", "roe", "roa", "rof", "returns"]])
print(fina_table_cloned[["end_date", "ca/liab", "cawi/liab", "roe", "roa", "rof", "returns"]])

current_assets_mv = current_market_value - (fina_table["total_cur_assets"].iloc[-1] - fina_table["total_liab"].iloc[-1])
current_returns = fina_table["rof"].iloc[-1] * fina_table["total_nca"].iloc[-1] / current_assets_mv
# rof # fixed assets
from termcolor import colored
print(f"{colored('[+]', 'green')} current returns : {current_returns:.2%}")

import matplotlib.pyplot as plt
def plot_bar(df: pd.DataFrame, col: str):
    data = df.copy()

    # 将end_date设置为索引
    data.set_index('end_date', inplace=True)
    data.sort_index(inplace=True)
    plt.figure(figsize=(18, 9))

    x = np.arange(0, len(data), 1)
    
    # 绘制柱状图
    plt.bar(x, data[col], width=0.5)
    plt.ylabel(col, fontsize=10)
    plt.xlabel('date', fontsize=10)

    # plt.tick_params(axis='x',length=0)
    from typing import Sequence
    xticks = [t.strftime(r'%Y-%m-%d') for t in data.index]
    plt.xticks(x, xticks, fontsize=10, rotation=45)
    plt.tick_params(axis='x',length=0)

    plt.grid(ls='--',alpha=0.8)

    plt.tight_layout()

    # 显示图形
    plt.show()

plot_bar(fina_table, "n_income_attr_p_ttm")
plot_bar(fina_table, "rof")
plot_bar(fina_table, "total_nca")