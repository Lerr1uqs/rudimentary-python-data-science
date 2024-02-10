# REF: https://www.wolai.com/stupidccl/3QzCwVcyRScvSt9nUugzQG#:~:text=show()-,RSI%E5%9B%A0%E5%AD%90%E9%A2%84%E5%A4%84%E7%90%86%2D%E6%8A%80%E6%9C%AF%E5%9B%A0%E5%AD%90,-%E4%B8%8B%E9%9D%A2%E5%B0%86%E4%BB%A5
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import tushare as ts

import warnings
warnings.simplefilter(action="ignore", category=Warning)


trading_data_2019 = pd.read_csv('../assets/2019_trading_data.csv')
# 中国平安
ZGPA = trading_data_2019.query("secucode=='601318.SH'").set_index('data_date')
ZGPA['returns'] = ZGPA['daily_return'].add(1).cumprod()
ZGPA['returns'].plot(figsize=(18,9))


def rsi(rolling_ser_chg: pd.Series):
    returns = 100 * rolling_ser_chg.add(1).cumprod()
    returns_diff = returns.diff(1).dropna()
    
    rs = sum([item for item in returns_diff if item > 0]) / sum([-item for item in returns_diff if item < 0])
    return 100 * (rs / (1 + rs))


# 要有14天能与前一天能diff出来 所以要15天 
# 这里没用价格去做rsi而是用回报率做rsi 就是变相复权了
# 这也就是价格指数化
ZGPA['rsi'] = ZGPA['daily_return'].rolling(15).apply(rsi)
ZGPA['rsi'].plot(figsize=(18, 9))
plt.show()