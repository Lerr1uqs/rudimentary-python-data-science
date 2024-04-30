import os 
import tushare as ts
import pandas as pd
import numpy as np
from typing import cast

TOKEN_PATH = os.path.expanduser('~/.tushare.token')
with open(TOKEN_PATH, 'r') as f:
    token = f.read().strip()
    ts.set_token(token)
    pro = ts.pro_api()

START = "20180401"
END   = "20240430"
TICKER = "000651.SZ" 
print("格力电器")
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
# data: pd.DataFrame = ts.pro_bar(ts_code=TICKER, adj='qfq', start_date=START, end_date=END)

# fina_col = ["ts_code", "end_date", "roe", "q_roe", "roe_yearly"]
# fina_ind = pro.fina_indicator(
#     ts_code=TICKER, 
#     start_date=START, 
#     end_date=END
# )[fina_col]

# fina_ind = fina_ind.drop_duplicates(subset='end_date') # 有多个end_date

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
        "total_liab",   # 负债合计
    ])
)

fina_table = pd.merge(income, balancesheet, on='end_date')
fina_table.drop_duplicates(subset='end_date', inplace=True)
# fina_table = fina_table[fina_table["end_date"].str.endswith("1231")] # 只估算每年最后一年 如果要滚动 就要加其他逻辑 因为净收入是每季度叠加的
fina_table.loc[:, "end_date"] = pd.to_datetime(fina_table["end_date"])
fina_table.sort_values("end_date", inplace=True)
fina_table.reset_index(drop=True, inplace=True)

fina_table["n_income_attr_p_diff"] = 0.0
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
factor.loc[:, "end_date"] = pd.to_datetime(factor["end_date"])
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

# (市值 - (现金 - 负债)) / 非流动性资产 = rof / 收益率
# 收益率 = rof * 非流动性资产 / (市值 - (现金 - 负债))
# assets_market_value : 去除市值中对负债和现金的估计
fina_table["assets_market_value"] = fina_table["total_mv"] - (fina_table["total_cur_assets"] - fina_table["total_liab"])
fina_table["returns"] = fina_table["rof"] * fina_table["total_nca"] / fina_table["assets_market_value"]
print(fina_table[["end_date", "rof", "returns"]])
# rof # fixed assets