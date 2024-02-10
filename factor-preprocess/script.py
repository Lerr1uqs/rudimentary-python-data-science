import pandas as pd

# 读取 HDF5 文件
file_path = '../assets/roe.h5'

data = pd.read_hdf(file_path)

data_2019_05_27 = data[data['data_date'] == '2019-05-27']

# 将数据写回 HDF5 文件，覆盖之前的数据
data_2019_05_27.to_hdf(file_path, key='2019-05-27', mode='w')
