# REF: https://www.wolai.com/stupidccl/3QzCwVcyRScvSt9nUugzQG
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import tushare as ts


trading_data_2019 = pd.read_csv('../assets/2019_trading_data.csv')
sub_trading_data = trading_data_2019.query("data_date=='2019-05-27'")
sub_trading_data['size'] = np.log(sub_trading_data['mv'])

def filter_extreme_by_sigma(series: pd.Series, n=3):
    # 计算均值
    mean = series.mean()
    # 计算方差
    std = series.std()
    # 计算上下限的值
    max_value = mean + n * std
    min_value = mean - n * std
    # clip: 修剪
    return np.clip(series, min_value, max_value)

def standard_normalize(series: pd.Series) -> pd.Series:
    std = series.std()
    mean = series.mean()
    return (series - mean) / std

sub_trading_data['size'] = filter_extreme_by_sigma(sub_trading_data['size'])
sub_trading_data['size'] = standard_normalize(sub_trading_data['size'])

def industry_and_size_neutralization(factor_df: pd.DataFrame, factor_name):
    # roe = β0 * ln(MktVal) + sum βi * industryi + varepsilon  
    result = sm.OLS(
        factor_df[factor_name], 
        factor_df[list(factor_df.ind_code.unique()) + ['size']], 
        hasconst=False
    ).fit()
    return result.resid

roe_factor = pd.read_hdf('../assets/roe.h5')
# 去极值
roe_factor['roe'] = np.clip(roe_factor['roe'], 0, 30)
# 标准化
roe_factor['roe'] = standard_normalize(roe_factor['roe'])


sub_trading_data = pd.concat([sub_trading_data, pd.get_dummies(sub_trading_data['ind_code'], dtype=int)], axis=1)
# merge对a和b中相同的key取交集 比如 a中有一行A,1 b中有一行A,2 merge后就是A,1,2 以A这一列的数据为key进行交集
# 这里的key是secucode 也就是给sub_trading_data每一行加上一列roe
roe_neuted_df = pd.merge(sub_trading_data, roe_factor[['secucode', 'roe']])
roe_neuted_df['roe_neuted'] = industry_and_size_neutralization(roe_neuted_df, 'roe')


roe_neuted_df['roe_neuted'].hist(bins=100, figsize=(18, 9))
plt.show()
