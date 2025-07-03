import os
import re
import time

import numpy as np
import pandas as pd

from scipy.stats import boxcox
from sklearn.preprocessing import OneHotEncoder

from data_dist_estimation import collect_statistics


'''
 特征提取
 需要增加周期化的时间特征来学习昼夜规律。
'''
def feature_extra(data_df):
    data_df['time'] = pd.to_datetime(data_df['time'], format='mixed')

    data_df['hour_sin'] = np.sin(2 * np.pi * data_df["time"].dt.hour / 24)
    data_df['hour_cos'] = np.cos(2 * np.pi * data_df["time"].dt.hour / 24)
    data_df['minute_sin'] = np.sin(2 * np.pi * data_df["time"].dt.minute / 60)
    data_df['minute_cos'] = np.cos(2 * np.pi * data_df["time"].dt.minute / 60)

    return data_df

def encoding(data_df):
    data_df.loc[data_df['sex'] == 'M', 'encoded_sex'] = 1
    data_df.loc[data_df['sex'] == 'F', 'encoded_sex'] = 0

    onehot_encoder = OneHotEncoder(categories=[[0, 1]], sparse_output=False)
    sex_onehot = onehot_encoder.fit_transform(data_df[['encoded_sex']])
    sex_onehot = pd.DataFrame(sex_onehot).rename(columns={0: 'sex_F', 1: 'sex_M'})

    data_df.loc[data_df['age'] == '18-29', 'encoded_age'] = 0
    data_df.loc[data_df['age'] == '30-37', 'encoded_age'] = 1
    data_df.loc[data_df['age'] == '38-52', 'encoded_age'] = 2
    data_df.loc[data_df['age'] == '53+', 'encoded_age'] = 3

    onehot_encoder = OneHotEncoder(categories=[[0, 1, 2, 3]], sparse_output=False)
    age_onehot = onehot_encoder.fit_transform(data_df[['encoded_age']])
    age_onehot = pd.DataFrame(age_onehot).rename(columns={0: 'age_18-29', 1: 'age_30-37',
                                                          2: 'age_38-52', 3: 'age_53+'})

    data_df.drop(['encoded_sex', 'encoded_age'], axis=1, inplace=True)
    onehot_encoded = pd.concat([sex_onehot, age_onehot], axis=1)
    data_df = pd.concat([data_df, onehot_encoded], axis=1)
    return data_df

# Robust Z-score
def normalization(data_series, column_median, column_iqr, alpha=1.0):
    data_series = alpha * (data_series - column_median) / (column_iqr + 1e-8)
    return data_series

# norm_MET_target 反归一化
def met_anti_norm(data, column_median, column_iqr, alpha=1.0):
    data = data * (column_iqr + 1e-8) / alpha + column_median
    return data


if __name__ == '__main__':
    dataset_path_csv_path = '../../output_and_result_files/dataset_paths.csv'

    # 读取 Metadata
    metadata_df = pd.read_csv('../../附件1/Metadata1.csv', low_memory=False)

    # 获取训练数据的所有 CSV 文件的路径
    dataset_path = pd.read_csv(dataset_path_csv_path)
    train_file_list = list(dataset_path.loc[dataset_path['type'] == 'train', 'file_path'])

    des_est_df = pd.read_csv(
        '../../dataset_distribution/original_data_distribution/train_data_descriptive_statistics.csv', index_col=0)
    x_acc_median = des_est_df.loc['x', 'median']
    x_acc_iqr = des_est_df.loc['x', 'iqr']

    y_acc_median = des_est_df.loc['y', 'median']
    y_acc_iqr = des_est_df.loc['y', 'iqr']

    z_acc_median = des_est_df.loc['z', 'median']
    z_acc_iqr = des_est_df.loc['z', 'iqr']

    acc_xy_l2_median = des_est_df.loc['acc_xy_L2', 'median']
    acc_xy_l2_iqr = des_est_df.loc['acc_xy_L2', 'iqr']

    acc_z_abs_median = des_est_df.loc['acc_z_abs', 'median']
    acc_z_abs_iqr = des_est_df.loc['acc_z_abs', 'iqr']

    met_median = des_est_df.loc['MET', 'median']
    met_iqr = des_est_df.loc['MET', 'iqr']


    # 定义正则表达式
    pattern = re.compile(r'^(P\d+)\.csv$')

    # 获取验证数据的所有 CSV 文件的路径
    val_file_list = list(dataset_path.loc[dataset_path['type'] == 'val', 'file_path'])

    # 获取测试数据的所有 CSV 文件的路径
    test_file_list = list(dataset_path.loc[dataset_path['type'] == 'test', 'file_path'])

    file_list_list = [train_file_list, val_file_list, test_file_list]

    print(f"开始进行归一化和特征编码")
    total_start_time = time.time()

    lambda_records = []

    # 处理训练集、验证集和测试集数据
    for i, file_list in enumerate(file_list_list):
        for j, file_path in enumerate(file_list):
            # 从文件路径中提取完整文件名
            filename = os.path.basename(file_path)

            # 使用正则查找匹配项
            match = pattern.search(filename)
            if match:
                extracted_part = match.group(1)  # 获取匹配的部分
            else:
                extracted_part = 'Unknown'

            print(f'开始归一化，共{len(file_list_list)}个数据集，在处理第{i+1}个数据集 /n '
                  f'数据集有{len(file_list)}个csv文件，在处理第{j+1}个csv文件,文件名{extracted_part}.csv')
            start_time = time.time()

            # 读取志愿者数据
            data_df = pd.read_csv(file_path)

            # 将 Metadata 的年龄和性别数据整理到一起
            data_df['age'] = metadata_df.loc[metadata_df['pid'] == extracted_part, 'age'].iloc[0]
            data_df['sex'] = metadata_df.loc[metadata_df['pid'] == extracted_part, 'sex'].iloc[0]

            data_df = feature_extra(data_df)

            # 特征编码
            data_df = encoding(data_df)

            # 对验证集和测试集的归一化也要使用训练集中的最大、最小值
            # Robust Z-Score 归一化，由 train_data_descriptive_statistics 中的数据推断，加速度数据分布近似对称分布，故直接归一化
            data_df['norm_x'] = normalization(data_df['x'], x_acc_median, x_acc_iqr)
            data_df['norm_y'] = normalization(data_df['y'], y_acc_median, y_acc_iqr, alpha=1.0/2.5)
            data_df['norm_z'] = normalization(data_df['z'], z_acc_median, z_acc_iqr, alpha=1.0/1.2)
            data_df['norm_acc_xy_L2'] = normalization(data_df['acc_xy_L2'], acc_xy_l2_median,
                                                      acc_xy_l2_iqr, alpha=1.0/4.2)
            data_df['norm_acc_z_abs'] = normalization(data_df['acc_z_abs'], acc_z_abs_median,
                                                      acc_z_abs_iqr, alpha=1.0/1.6)
            data_df['norm_MET'] = normalization(data_df['MET'], met_median, met_iqr, alpha=1.0/3.0)
            data_df['norm_MET_target'] = normalization(data_df['MET'], met_median, met_iqr)

            # 对于数据分布有较强的正偏态(skew > 1)或较高的峰度(kurtosis > 10)（右尾长），故进行Box-Cox变换后，再归一化，避免极值主导缩放
            boxcox_acc_l2, acc_l2_lambda_opt = boxcox(data_df['acc_L2'].values)
            min_clip = np.quantile(boxcox_acc_l2, 0.05)
            max_clip = np.quantile(boxcox_acc_l2, 0.95)
            data_df['boxcox_acc_L2'] = np.clip(boxcox_acc_l2, a_min=min_clip, a_max=max_clip)

            lambda_records.append({'file': filename, 'acc_l2_lambda': acc_l2_lambda_opt})

            # 构造离散时间段
            data_df['time'] = pd.to_datetime(data_df['time'], format='mixed')
            hour = data_df['time'].dt.hour + data_df['time'].dt.minute / 60

            data_df['is_midnight'] = hour.between(0, 5, inclusive="left").astype(int)
            data_df['is_weekend'] = data_df['time'].dt.weekday.between(5, 6).astype(int)

            data_df.sort_values(by="time", inplace=True, ascending=True)

            data_df.to_csv('../../附件1/' + extracted_part + '.csv', index=False)

            end_time = time.time()
            one_file_time = end_time - start_time
            print(f'处理{extracted_part}文件用了{one_file_time / 60:.2f}分钟')

    pd.DataFrame(lambda_records).to_csv('../../附件1/acc_l2_lambdas.csv', index=False)


    # 获得 acc_L2 Box-Cox变换后的统计量
    statis_dict = collect_statistics(['boxcox_acc_L2'], train_file_list)
    if 'boxcox_acc_L2' not in des_est_df.index:
        des_est_df.loc[len(des_est_df)] = [statis_dict['boxcox_acc_L2']['count'], statis_dict['boxcox_acc_L2']['trimmed_mean'],
                                           statis_dict['boxcox_acc_L2']['mean'], statis_dict['boxcox_acc_L2']['std'],
                                           statis_dict['boxcox_acc_L2']['min'], statis_dict['boxcox_acc_L2']['q1'],
                                           statis_dict['boxcox_acc_L2']['median'], statis_dict['boxcox_acc_L2']['q3'],
                                           statis_dict['boxcox_acc_L2']['max'], statis_dict['boxcox_acc_L2']['iqr'],
                                           statis_dict['boxcox_acc_L2']['skew'], statis_dict['boxcox_acc_L2']['kurtosis'],
                                           statis_dict['boxcox_acc_L2']['outlier_lower_3sigma'], statis_dict['boxcox_acc_L2']['outlier_upper_3sigma'],
                                           statis_dict['boxcox_acc_L2']['outlier_lower_quantile'], statis_dict['boxcox_acc_L2']['outlier_upper_quantile']]
        des_est_df.index = ['x', 'y', 'z', 'MET', 'acc_L2', 'acc_xy_L2', 'acc_z_abs', 'boxcox_acc_L2']

    else:
        des_est_df.loc['boxcox_acc_L2'] = [statis_dict['boxcox_acc_L2']['count'], statis_dict['boxcox_acc_L2']['trimmed_mean'],
                                           statis_dict['boxcox_acc_L2']['mean'], statis_dict['boxcox_acc_L2']['std'],
                                           statis_dict['boxcox_acc_L2']['min'], statis_dict['boxcox_acc_L2']['q1'],
                                           statis_dict['boxcox_acc_L2']['median'], statis_dict['boxcox_acc_L2']['q3'],
                                           statis_dict['boxcox_acc_L2']['max'], statis_dict['boxcox_acc_L2']['iqr'],
                                           statis_dict['boxcox_acc_L2']['skew'], statis_dict['boxcox_acc_L2']['kurtosis'],
                                           statis_dict['boxcox_acc_L2']['outlier_lower_3sigma'], statis_dict['boxcox_acc_L2']['outlier_upper_3sigma'],
                                           statis_dict['boxcox_acc_L2']['outlier_lower_quantile'], statis_dict['boxcox_acc_L2']['outlier_upper_quantile']]

    des_est_df.to_csv('../../dataset_distribution/original_data_distribution/train_data_descriptive_statistics.csv')

    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    print(f'数据归一化完成！共使用{total_time / 60:.2f}分钟！')
