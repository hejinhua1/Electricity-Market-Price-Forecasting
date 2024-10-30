import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler

# 定义数据清洗函数
def data_replacing(data, column_name):
    """
    数据清洗函数
    :param data: 需要清洗的数据
    :param column_name: 需要清洗的列名
    :return: 清洗后的数据
    """

    # 与均值的差值超过3倍标准差的用均值代替
    mean = data[column_name].mean()
    std = data[column_name].std()
    data[column_name] = np.where(abs(data[column_name] - mean) > 3 * std, mean, data[column_name])
    return data
# 定义数据填充函数
def data_filling(data, column_name):
    """
    数据填充函数
    :param data: 需要填充的数据
    :param column_name: 需要填充的列名
    :return: 填充后的数据
    """
    # 用前一个值填充
    data[column_name] = data[column_name].fillna(method='ffill')
    return data




if __name__ == '__main__':


    # # Load the data from the uploaded file
    # file_path = '../data/gansu_quansheng.xlsx'
    # data = pd.read_excel(file_path)
    #
    # # 去掉第一列
    # data_cleaned = data.drop(columns=['Unnamed: 0'])
    # # 时间列转换为时间格式
    # data_cleaned['date'] = pd.to_datetime(data_cleaned['date'])
    # # 去掉‘wind_power_forecast’列
    # data_cleaned = data_cleaned.drop(columns=['wind_power_forecast'])
    # # 去掉‘photovoltaic_power_forecast’列
    # data_cleaned = data_cleaned.drop(columns=['photovoltaic_power_forecast'])
    # # 去掉‘realtime_clearing_price’列
    # data_cleaned = data_cleaned.drop(columns=['realtime_clearing_price'])
    #
    # # 检查数据缺失情况，生成从2024年1月1日到2024年10月23日间隔15分钟的时间序列
    # date_range = pd.date_range(start='2024-01-01', end='2024-10-24', freq='15T')
    # # 检查数据缺失情况
    # # missing_date = [date for date in date_range if date not in data_cleaned['date'].values]
    #
    # # 进行数据清洗，超过3倍标准差的用均值代替
    # data_columns = data_cleaned.columns
    # print(data_columns)
    # replacing_columns = ['provincial_load_forecast', 'tie_line_load_forecast', 'total_power_forecast',
    #                      'new_energy_power_forecast', 'hydraulic_power_forecast', 'non_market_power_forecast']
    # for column_name in replacing_columns:
    #     data_cleaned = data_replacing(data_cleaned, column_name)
    # # 进行数据填充，用前一个值填充
    # fill_columns = ['provincial_load_forecast', 'tie_line_load_forecast', 'total_power_forecast',
    #                 'new_energy_power_forecast', 'hydraulic_power_forecast', 'non_market_power_forecast',
    #                 'dayahead_clearing_price']
    # # for column_name in fill_columns:
    # #     data_cleaned = data_filling(data_cleaned, column_name)
    # # 检查每一列数据缺失情况
    # missing_data = data_cleaned.isnull().sum()
    # print(missing_data)

    # # 画图每一列数据，并保存600dpi的图片
    # for column_name in fill_columns:
    #     plt.figure(figsize=(12, 6))
    #     plt.plot(data_cleaned['date'], data_cleaned[column_name])
    #     plt.title(column_name)
    #     plt.xlabel('date')
    #     plt.ylabel(column_name)
    #     plt.savefig(f'../data/pic/{column_name}.png', dpi=600)
    #     plt.close()
    #
    # # 单独画出‘dayahead_clearing_price’列的每一天的数据, x轴是00:00到23:45,15分钟一个点，所有天数画在同一张图上
    # dayahead_clearing_price = data_cleaned['dayahead_clearing_price']
    # dayahead_clearing_price = dayahead_clearing_price.reset_index(drop=True)
    # dayahead_clearing_price = dayahead_clearing_price.values
    # dayahead_clearing_price = dayahead_clearing_price.reshape(-1, 96)
    # plt.figure(figsize=(12, 6))
    # for i in range(len(dayahead_clearing_price)):
    #     plt.plot(dayahead_clearing_price[i])
    # plt.title('dayahead_clearing_price')
    # plt.xlabel('time')
    # plt.ylabel('dayahead_clearing_price')
    # plt.savefig(f'../data/pic/dayahead_clearing_price_all.png', dpi=600)






    # # 保存清洗后的数据为feather格式
    # data_cleaned.reset_index(drop=True).to_feather('../data/full_data.feather')
    # # 拆分数据集
    # # 训练集为2024年9月1日之前的数据
    # data_train = data_cleaned[data_cleaned['date'] < '2024-09-01']
    # # 验证集为2024年9月1日至2024年10月1日的数据
    # data_val = data_cleaned[(data_cleaned['date'] >= '2024-09-01') & (data_cleaned['date'] < '2024-10-01')]
    # # 测试集为2024年10月1日之后的数据
    # data_test = data_cleaned[data_cleaned['date'] >= '2024-10-01']
    # # 保存拆分后的数据集为feather格式
    # data_train.reset_index(drop=True).to_feather('../data/train_data.feather')
    # data_val.reset_index(drop=True).to_feather('../data/val_data.feather')
    # data_test.reset_index(drop=True).to_feather('../data/test_data.feather')

    # 读取数据
    data_full = pd.read_feather('../data/full_data.feather')

    # 选择需要归一化的列
    columns_to_normalize_X = ['time_order', 'provincial_load_forecast', 'tie_line_load_forecast',
                            'total_power_forecast', 'new_energy_power_forecast',
                            'hydraulic_power_forecast', 'non_market_power_forecast']
    columns_to_normalize_Y = ['dayahead_clearing_price']
    # 创建缩放器
    scalerX = StandardScaler()
    scalerY = StandardScaler()

    # 应用标准化
    data_full[columns_to_normalize_X] = scalerX.fit_transform(data_full[columns_to_normalize_X])
    data_full[columns_to_normalize_Y] = scalerY.fit_transform(data_full[columns_to_normalize_Y])
    # 保存 scaler
    joblib.dump(scalerX, '../data/scalerX.joblib')
    joblib.dump(scalerY, '../data/scalerY.joblib')


    # # 加载 scaler
    scalerX = joblib.load('../data/scalerX.joblib')
    scalerY = joblib.load('../data/scalerY.joblib')
    data_train = pd.read_feather('../data/train_data.feather')
    data_val = pd.read_feather('../data/val_data.feather')
    data_test = pd.read_feather('../data/test_data.feather')

    # 对上面的数据集进行归一化
    data_train[columns_to_normalize_X] = scalerX.transform(data_train[columns_to_normalize_X])
    data_val[columns_to_normalize_X] = scalerX.transform(data_val[columns_to_normalize_X])
    data_test[columns_to_normalize_X] = scalerX.transform(data_test[columns_to_normalize_X])

    data_train[columns_to_normalize_Y] = scalerY.transform(data_train[columns_to_normalize_Y])
    data_val[columns_to_normalize_Y] = scalerY.transform(data_val[columns_to_normalize_Y])
    data_test[columns_to_normalize_Y] = scalerY.transform(data_test[columns_to_normalize_Y])

    # 对数据进行反归一化
    data_train[columns_to_normalize_X] = scalerX.inverse_transform(data_train[columns_to_normalize_X])
    data_val[columns_to_normalize_X] = scalerX.inverse_transform(data_val[columns_to_normalize_X])
    data_test[columns_to_normalize_X] = scalerX.inverse_transform(data_test[columns_to_normalize_X])

    data_train[columns_to_normalize_Y] = scalerY.inverse_transform(data_train[columns_to_normalize_Y])
    data_val[columns_to_normalize_Y] = scalerY.inverse_transform(data_val[columns_to_normalize_Y])
    data_test[columns_to_normalize_Y] = scalerY.inverse_transform(data_test[columns_to_normalize_Y])








