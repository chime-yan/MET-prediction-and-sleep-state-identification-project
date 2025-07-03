"""
对于 MET 和 加速度 的异常值，在正常的取值范围中，可能是状态变化的节点，故保留，并添加对应特征的异常指示特征 is_outlier_feature
"""

import glob
import os
import re

import pandas as pd

from data_outlier_detection import outlier_detection


if __name__ == '__main__':
    # 获取文件夹下所有 CSV 文件的路径
    # file_list = glob.glob('../../附件1/*.csv')
    file_list = glob.glob('../../附件2/*.csv')

    # 定义正则表达式
    pattern = re.compile(r'^(P\d+)\.csv$')

    # 过滤出符合条件的csv文件
    file_list = [f for f in file_list if pattern.match(os.path.basename(f))]
    print(f"共发现 {len(file_list)} 个符合的 CSV 文件")

    abnormal_df_all = pd.DataFrame()

    stats = pd.read_csv('../../dataset_distribution/original_data_distribution/all_data_descriptive_statistics.csv', index_col=0)

    for i, file_path in enumerate(file_list):
        # 从文件路径中提取完整文件名
        filename = os.path.basename(file_path)

        # 使用正则查找匹配项
        match = pattern.search(filename)
        if match:
            extracted_part = match.group(1)  # 获取匹配的部分
        else:
            extracted_part = 'Unknown'

        print(f'第{i + 1}个/共{len(file_list)}个，开始处理{extracted_part}')

        # 读取志愿者数据
        data_df = pd.read_csv(file_path, low_memory=False)

        # data_df['is_outlier_MET'] = 0
        data_df['is_outlier_x'] = 0
        data_df['is_outlier_y'] = 0
        data_df['is_outlier_z'] = 0

        # MET为右偏态分布(高峰重尾),使用分位数法检测异常值
        # met_abnormal_index = outlier_detection(data_df, 'MET',
        #                                        stats.loc['MET', 'outlier_lower_quantile'],
        #                                        stats.loc['MET', 'outlier_upper_quantile'])
        # data_df.loc[met_abnormal_index, 'is_outlier_MET'] = 1

        # 加速度近似正态分布，使用3sigma原则检测异常值
        x_abnormal_index = outlier_detection(data_df, 'x',
                                             stats.loc['x', 'outlier_lower_3sigma'],
                                             stats.loc['x', 'outlier_upper_3sigma'])
        data_df.loc[x_abnormal_index, 'is_outlier_x'] = 1

        y_abnormal_index = outlier_detection(data_df, 'y',
                                             stats.loc['y', 'outlier_lower_3sigma'],
                                             stats.loc['y', 'outlier_upper_3sigma'])
        data_df.loc[y_abnormal_index, 'is_outlier_y'] = 1

        z_abnormal_index = outlier_detection(data_df, 'z',
                                             stats.loc['z', 'outlier_lower_3sigma'],
                                             stats.loc['z', 'outlier_upper_3sigma'])
        data_df.loc[z_abnormal_index, 'is_outlier_z'] = 1

        # data_df.to_csv('../../附件1/' + extracted_part + '.csv', index=False)
        data_df.to_csv('../../附件2/' + extracted_part + '.csv', index=False)

    print("处理异常值完毕")
