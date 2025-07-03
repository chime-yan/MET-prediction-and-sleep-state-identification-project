import glob
import os
import re

import pandas as pd


def outlier_detection(data_df, column, lower_bound, upper_bound):
    # 获取并返回异常值对应的索引
    outlier_indices = data_df[(data_df[column] < lower_bound) | (data_df[column] > upper_bound)].index
    return outlier_indices


if __name__ == '__main__':
    # 获取文件夹下所有 CSV 文件的路径
    file_list = glob.glob('../../附件1/*.csv')

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


        # 数据特征异常检测
        overall_count = data_df.shape[0]

        # 加速度异常检测，加速度近似正态分布，使用3sigma原则检测异常值
        x_abnormal_index = outlier_detection(data_df, 'x',
                                            stats.loc['x', 'outlier_lower_3sigma'],
                                             stats.loc['x', 'outlier_upper_3sigma'])

        y_abnormal_index = outlier_detection(data_df, 'y',
                                            stats.loc['y', 'outlier_lower_3sigma'],
                                             stats.loc['y', 'outlier_upper_3sigma'])

        z_abnormal_index = outlier_detection(data_df, 'z',
                                            stats.loc['z', 'outlier_lower_3sigma'],
                                             stats.loc['z', 'outlier_upper_3sigma'])

        # MET 异常检测，MET为右偏态分布(高峰重尾)，使用分位数法检测异常值
        met_abnormal_index = outlier_detection(data_df, 'MET',
                                            stats.loc['MET', 'outlier_lower_quantile'],
                                             stats.loc['MET', 'outlier_upper_quantile'])

        x_abnormal_count = data_df.loc[x_abnormal_index, 'x'].count()
        y_abnormal_count = data_df.loc[y_abnormal_index, 'y'].count()
        z_abnormal_count = data_df.loc[z_abnormal_index, 'z'].count()
        met_abnormal_count = data_df.loc[met_abnormal_index, 'x'].count()
        abnormal_df = pd.DataFrame({
            'pid': [extracted_part],
            'x': [x_abnormal_count],
            'y': [y_abnormal_count],
            'z': [z_abnormal_count],
            'MET': [met_abnormal_count],
            '所有数据的总数据量': [overall_count],
            'x异常数据占比': [round(x_abnormal_count / overall_count, 4)],
            'y异常数据占比': [round(y_abnormal_count / overall_count, 4)],
            'z异常数据占比': [round(z_abnormal_count / overall_count, 4)],
            'MET异常数据占比': [round(met_abnormal_count / overall_count, 4)]
        })
        abnormal_df_all = pd.concat([abnormal_df_all, abnormal_df], ignore_index=True)

    os.makedirs('../../data_glance_results', exist_ok=True)
    abnormal_df_all.to_excel('../../data_glance_results/data_abnormal.xlsx', index=False)
    print('数据异常值检测完成！')
