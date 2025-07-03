import os
import re
import time

import numpy as np
import pandas as pd

from data_dist_estimation import collect_statistics
from data_outlier_detection import outlier_detection


if __name__ == '__main__':
    dataset_path_csv_path = '../../output_and_result_files/dataset_paths.csv'

    # 获取训练数据的所有 CSV 文件的路径
    dataset_path = pd.read_csv(dataset_path_csv_path)
    train_file_list = list(dataset_path.loc[dataset_path['type'] == 'train', 'file_path'])

    # 定义正则表达式
    pattern = re.compile(r'^(P\d+)\.csv$')

    # 获取验证数据的所有 CSV 文件的路径
    dataset_path = pd.read_csv(dataset_path_csv_path)
    val_file_list = list(dataset_path.loc[dataset_path['type'] == 'val', 'file_path'])

    # 获取测试数据的所有 CSV 文件的路径
    dataset_path = pd.read_csv(dataset_path_csv_path)
    test_file_list = list(dataset_path.loc[dataset_path['type'] == 'test', 'file_path'])

    file_list_list = [train_file_list, val_file_list, test_file_list]

    print(f"开始进行归一化...")
    total_start_time = time.time()

    stats = pd.read_csv('../../dataset_distribution/original_data_distribution/train_data_descriptive_statistics.csv', index_col=0)

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

            print(f'开始归一化，共{len(file_list_list)}个数据集，在处理第{i + 1}个数据集 /n '
                  f'数据集有{len(file_list)}个csv文件，在处理第{j + 1}个csv文件,文件名{extracted_part}.csv')
            start_time = time.time()

            # 读取志愿者数据
            data_df = pd.read_csv(file_path)

            data_df['is_outlier_accL2_MET'] = 0

            # 根据 train_data_descriptive_statistics.csv 可知，特征 accL2_MET 是右偏的
            data_df['log_accL2_MET'] = data_df['accL2_MET'].transform(lambda x: np.log(x + 0.1))

            # accL2_MET为右偏态分布(高峰重尾),使用分位数法检测异常值
            accL2_MET_abnormal_index = outlier_detection(data_df, 'accL2_MET',
                                                      stats.loc['accL2_MET', 'outlier_lower_quantile'],
                                                      stats.loc['accL2_MET', 'outlier_upper_quantile'])
            data_df.loc[accL2_MET_abnormal_index, 'is_outlier_accL2_MET'] = 1

            data_df.to_csv('../../附件1/' + extracted_part + '.csv', index=False)

            end_time = time.time()
            one_file_time = end_time - start_time
            print(f'处理{extracted_part}文件用了{one_file_time / 60:.2f}分钟')


    # 获得以上特征对数变换后的统计量
    des_est_df = pd.read_csv(
        '../../dataset_distribution/original_data_distribution/train_data_descriptive_statistics.csv', index_col=0)

    statis_dict = collect_statistics(['log_accL2_MET'], train_file_list)

    if 'log_accL2_MET' not in des_est_df.index:
        des_est_df.loc[len(des_est_df)] = [statis_dict['log_accL2_MET']['count'], statis_dict['log_accL2_MET']['trimmed_mean'],
                                           statis_dict['log_accL2_MET']['mean'], statis_dict['log_accL2_MET']['std'],
                                           statis_dict['log_accL2_MET']['min'], statis_dict['log_accL2_MET']['q1'],
                                           statis_dict['log_accL2_MET']['median'], statis_dict['log_accL2_MET']['q3'],
                                           statis_dict['log_accL2_MET']['max'], statis_dict['log_accL2_MET']['iqr'],
                                           statis_dict['log_accL2_MET']['skew'], statis_dict['log_accL2_MET']['kurtosis'],
                                           statis_dict['log_accL2_MET']['outlier_lower_3sigma'],
                                           statis_dict['log_accL2_MET']['outlier_upper_3sigma'],
                                           statis_dict['log_accL2_MET']['outlier_lower_quantile'],
                                           statis_dict['log_accL2_MET']['outlier_upper_quantile']
                                           ]
        des_est_df.index = ['x', 'y', 'z', 'MET', 'acc_L2', 'acc_xy_L2', 'acc_z_abs', 'boxcox_acc_L2',
                            'accL2_MET', 'log_accL2_MET']

    else:
        des_est_df.loc['log_accL2_MET'] = [statis_dict['log_accL2_MET']['count'], statis_dict['log_accL2_MET']['trimmed_mean'],
                                      statis_dict['log_accL2_MET']['mean'], statis_dict['log_accL2_MET']['std'],
                                      statis_dict['log_accL2_MET']['min'], statis_dict['log_accL2_MET']['q1'],
                                      statis_dict['log_accL2_MET']['median'], statis_dict['log_accL2_MET']['q3'],
                                      statis_dict['log_accL2_MET']['max'], statis_dict['log_accL2_MET']['iqr'],
                                      statis_dict['log_accL2_MET']['skew'], statis_dict['log_accL2_MET']['kurtosis'],
                                               statis_dict['log_accL2_MET']['outlier_lower_3sigma'],
                                               statis_dict['log_accL2_MET']['outlier_upper_3sigma'],
                                               statis_dict['log_accL2_MET']['outlier_lower_quantile'],
                                               statis_dict['log_accL2_MET']['outlier_upper_quantile']
                                               ]

    des_est_df.to_csv('../../dataset_distribution/original_data_distribution/train_data_descriptive_statistics.csv')

    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    print(f'获取log数据统计量完成！共使用{total_time / 60:.2f}分钟！')
