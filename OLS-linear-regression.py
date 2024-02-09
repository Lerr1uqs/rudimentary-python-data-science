import pandas as pd
import statsmodels.api as smapi
import numpy as np

np.random.seed(1)

# 创建一个简化的DataFrame作为例子
data = {
    'factor': pd.array(np.linspace(1, 10, 100)),  # 随机生成因子数据
    'ind_code': pd.array(np.random.choice(['A', 'B', 'C', 'D'], size=100)),  # 行业代码
}

df = pd.DataFrame(data)

# 使用pd.get_dummies创建行业虚拟变量
dummy_variables = pd.get_dummies(df['ind_code'], dtype=float)

# 将虚拟变量和因子合并到一个DataFrame
df = pd.concat([df, dummy_variables], axis=1)
'''
      factor ind_code    A    B    C    D
0        1.0        B  0.0  1.0  0.0  0.0
1   1.090909        D  0.0  0.0  0.0  1.0
2   1.181818        A  1.0  0.0  0.0  0.0
3   1.272727        A  1.0  0.0  0.0  0.0
4   1.363636        D  0.0  0.0  0.0  1.0
..       ...      ...  ...  ...  ...  ...
95  9.636364        C  0.0  0.0  1.0  0.0
96  9.727273        D  0.0  0.0  0.0  1.0
97  9.818182        B  0.0  1.0  0.0  0.0
98  9.909091        A  1.0  0.0  0.0  0.0
99      10.0        D  0.0  0.0  0.0  1.0
'''
print(df)

# 定义要中性化的因子名称
factor_name = 'factor'

# 使用行业虚拟变量进行中性化
residuals = smapi.OLS(
    y = df[factor_name].astype(float), 
    X = df[list(df.ind_code.unique())]
).fit().resid

# 将中性化后的残差添加回DataFrame
df['neutralized-factor'] = residuals

# 打印结果
print(df.head())
