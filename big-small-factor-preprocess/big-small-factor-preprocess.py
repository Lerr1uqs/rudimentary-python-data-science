import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
# REF: https://www.wolai.com/stupidccl/3QzCwVcyRScvSt9nUugzQG
# warnings.simplefilter(action="ignore", category=Warning)


trading_data_2019 = pd.read_csv('2019_trading_data.csv')
print(trading_data_2019)
sub_trading_data = trading_data_2019.query("data_date=='2019-05-27'")
sub_trading_data['mv'].hist(bins=100, figsize=(18, 9))
plt.show()
# 这样展示出来的结果为小市值偏多 数据集中 需要log使其分布均匀化
sub_trading_data['log-mv'] = np.log(sub_trading_data['mv'])
sub_trading_data['log-mv'].hist(bins=100, figsize=(18, 9))
plt.show()

'''
正偏度（Skewness > 0）： 表示数据分布向右倾斜。这意味着数据的尾部朝右边延伸，数据分布的右侧比左侧更长。通常，正偏度意味着数据中可能存在较大的正值（右侧的尾部）。

负偏度（Skewness < 0）： 表示数据分布向左倾斜。这意味着数据的尾部朝左边延伸，数据分布的左侧比右侧更长。通常，负偏度意味着数据中可能存在较大的负值（左侧的尾部）。

偏度为零（Skewness = 0）： 表示数据分布相对对称。在这种情况下，数据集的左侧和右侧的尾部长度相似，没有明显的偏斜。
'''
sub_trading_data["log-mv"].skew() # 1.2637189510611557
# 结果是很右偏的分布 有很多大市值的股票造成的

#--------------------------------------------------------------
# 市值对它们收益率的影响有边界递减效应 我们可以通过几个办法去排除
# 1. 标准差
def filter_extreme_by_sigma(series, n=3):
    # 计算均值
    mean = series.mean()
    # 计算方差
    std = series.std()
    # 计算上下限的值
    max_value = mean + n * std
    min_value = mean - n * std
    return np.clip(series, min_value, max_value)

sub_trading_data['size_3sigma'] = filter_extreme_by_sigma(sub_trading_data['log-mv'])
sub_trading_data['size_3sigma'].skew()
sub_trading_data['size_3sigma'].hist(bins=100, figsize=(18, 9))
plt.show()

# 0.9858182883295862
sub_trading_data["size_3sigma"].skew() 

# 2. MAD 绝对中位 中位±绝对中位
# 3. 百分位法 剔除一个百分位范围之外的所有值

# 效果最好的反而是MAD

# -------------------------------------------------------------
# 标准化
# 使得因子之间可以比较 去除量纲
# z-score
def zscore(series: pd.DataFrame):
    mean = series.mean()
    std = series.std()
    return (series - mean) / std

# 另外还有最大值和最小值标准化 (X - min) / (max - min)
sub_trading_data['size_3sigma_std'] = zscore(sub_trading_data['size_3sigma'])
sub_trading_data['size_3sigma_std'].hist(bins=100, figsize=(18, 9))
plt.show()# 这里只会改变横轴

# --------------------------------------------------------------
# 行业中性化 按照各个行业去比较市值因子
import statsmodels.api as smapi
# TODO: 这部分搞不明白 

def industry_neutralization(factor_df: pd.DataFrame, factor_name):
    # 将行业变成哑变量后再作为解释变量，而需要中性化的因子则作为被解释变量进行回归。回归的残差项就是被中性化后的因子值
    # 如果用行业哑变量来解释因子值，那么不能被行业解释的那一部分就是剔除了行业因素后的因子值。
    # 而剔除了行业因素的因子值就是回归模型的残差项，这也正是我们希望获得的行业中性化之后的结果

    # factor_df.ind_code.unique() 这里就是去重 取出所有的行业代码
    # Ordinary Least Squares
    '''
    x = np.random.rand(100)
    y = 2 * x + 1 + np.random.randn(100)  # 生成一个简单的线性关系，加上噪声
    
    # 添加截距项
    X = sm.add_constant(x)
    
    # 创建模型
    model = sm.OLS(y, X)
    '''
    result = smapi.OLS(
        factor_df[factor_name],  # 因变量（被解释变量）也就是实验观测值
        factor_df[list(factor_df.ind_code.unique())] # 自变量（解释变量） 自变量的观测值
    ).fit()
    # y = β0 + inda * β1 + indb * β2 + indc * β3 + varepsilon
    # 这里返回的是一个数组 代表每一个ind_code列中为1的元素对
    # result.summary()

    # 残差就是每个点到回归线的垂直距离 是一个size=x.shape的数组
    # 残差能当做标准差的原因是 残差就是点偏离期望(线性方程)的距离 和标准差的意义一样
    return result.resid

# pd.get_dummies: 分类数据转换为二进制矩阵
'''
# 创建包含分类数据的 DataFrame
data = {'Category': ['A', 'B', 'A', 'C', 'B']}
df = pd.DataFrame(data)

# 使用 get_dummies 进行独热编码
df_encoded = pd.get_dummies(df['Category'], prefix='Category')

# 将独热编码的结果添加回原始 DataFrame
df = pd.concat([df, df_encoded], axis=1)

  Category  Category_A  Category_B  Category_C
0        A           1           0           0
1        B           0           1           0
2        A           1           0           0
3        C           0           0           1
4        B           0           1           0
'''
sub_trading_data = pd.concat(
    [sub_trading_data, pd.get_dummies(sub_trading_data['ind_code'])], 
    axis=1
)
# 本质上就是把某一列的类中的元素扩散成多个列 并用flag标记
sub_trading_data['size_factor_neuted'] = industry_neutralization(sub_trading_data, 'size_3sigma_std')
sub_trading_data[['data_date', 'secucode', 'ind_code', 'size_3sigma_std', 'size_factor_neuted']]

sub_trading_data[['size_3sigma_std', 'size_factor_neuted']].hist(bins=100,figsize=(18, 9))
plt.show()
