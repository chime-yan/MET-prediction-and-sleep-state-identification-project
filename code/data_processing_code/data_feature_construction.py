import glob
import os
import re
import time

import numpy as np
import pandas as pd

from data_dist_estimation import collect_statistics


if __name__ == '__main__':
    total_start_time = time.time()

    # 获取文件夹下所有 CSV 文件的路径
    # file_list = glob.glob('../../附件1/*.csv')
    file_list = glob.glob('../../附件2/*.csv')

    # 定义正则表达式
    pattern = re.compile(r'^(P\d+)\.csv$')

    # 过滤出符合条件的csv文件
    file_list = [f for f in file_list if pattern.match(os.path.basename(f))]
    print(f"共发现 {len(file_list)} 个符合的 CSV 文件")

    abnormal_df_all = pd.DataFrame()

    des_stat = pd.read_csv('../../dataset_distribution/original_data_distribution/all_data_descriptive_statistics.csv', index_col=0)

    for i, file_path in enumerate(file_list):
        start_time = time.time()

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

        # 构造加速度的欧几里得范数 (L2 范数)，作为总体运动强度
        data_df['acc_L2'] = (
            np.sqrt(data_df['x'] ** 2 + data_df['y'] ** 2
                    + data_df['z'] ** 2))

        # 构造x, y方向的L2范数，作为水平方向运动强度
        data_df['acc_xy_L2'] = np.sqrt(data_df['x'] ** 2 + data_df['y'] ** 2)

        # 对z方向的加速度取绝对值，作为垂直方向运动强度
        data_df['acc_z_abs'] = np.abs(data_df['z'])

        # data_df.to_csv('../../附件1/' + extracted_part + '.csv', index=False)
        data_df.to_csv('../../附件2/' + extracted_part + '.csv', index=False)

        end_time = time.time()
        one_file_time = end_time - start_time
        print(f'{extracted_part}文件处理完成，用了{one_file_time / 60:.2f}分钟')

    #
    # stats = collect_statistics(['acc_L2', 'acc_xy_L2', 'acc_z_abs'], file_list)
    #
    # if 'acc_L2' not in des_stat.index:
    #     des_stat.loc[len(des_stat)] = [stats['acc_L2']['count'], stats['acc_L2']['trimmed_mean'], stats['acc_L2']['mean'],
    #                                 stats['acc_L2']['std'], stats['acc_L2']['min'],
    #                                 stats['acc_L2']['q1'], stats['acc_L2']['median'], stats['acc_L2']['q3'],
    #                                 stats['acc_L2']['max'], stats['acc_L2']['iqr'],
    #                                 stats['acc_L2']['skew'], stats['acc_L2']['kurtosis'],
    #                                 stats['acc_L2']['outlier_lower_3sigma'], stats['acc_L2']['outlier_upper_3sigma'],
    #                                 stats['acc_L2']['outlier_lower_quantile'], stats['acc_L2']['outlier_upper_quantile']]
    #
    #     des_stat.loc[len(des_stat)] = [stats['acc_xy_L2']['count'], stats['acc_xy_L2']['trimmed_mean'],
    #                                    stats['acc_xy_L2']['mean'],
    #                                    stats['acc_xy_L2']['std'], stats['acc_xy_L2']['min'],
    #                                    stats['acc_xy_L2']['q1'], stats['acc_xy_L2']['median'], stats['acc_xy_L2']['q3'],
    #                                    stats['acc_xy_L2']['max'], stats['acc_xy_L2']['iqr'],
    #                                    stats['acc_xy_L2']['skew'], stats['acc_xy_L2']['kurtosis'],
    #                                    stats['acc_xy_L2']['outlier_lower_3sigma'], stats['acc_xy_L2']['outlier_upper_3sigma'],
    #                                    stats['acc_xy_L2']['outlier_lower_quantile'],
    #                                    stats['acc_xy_L2']['outlier_upper_quantile']]
    #
    #     des_stat.loc[len(des_stat)] = [stats['acc_z_abs']['count'], stats['acc_z_abs']['trimmed_mean'],
    #                                    stats['acc_z_abs']['mean'],
    #                                    stats['acc_z_abs']['std'], stats['acc_z_abs']['min'],
    #                                    stats['acc_z_abs']['q1'], stats['acc_z_abs']['median'], stats['acc_z_abs']['q3'],
    #                                    stats['acc_z_abs']['max'], stats['acc_z_abs']['iqr'],
    #                                    stats['acc_z_abs']['skew'], stats['acc_z_abs']['kurtosis'],
    #                                    stats['acc_z_abs']['outlier_lower_3sigma'], stats['acc_z_abs']['outlier_upper_3sigma'],
    #                                    stats['acc_z_abs']['outlier_lower_quantile'],
    #                                    stats['acc_z_abs']['outlier_upper_quantile']]
    #
    #     des_stat.index = ['x', 'y', 'z', 'MET', 'acc_L2', 'acc_xy_L2', 'acc_z_abs']
    #
    # else:
    #     des_stat.loc['acc_L2'] = [stats['acc_L2']['count'], stats['acc_L2']['trimmed_mean'],
    #                                    stats['acc_L2']['mean'],
    #                                    stats['acc_L2']['std'], stats['acc_L2']['min'],
    #                                    stats['acc_L2']['q1'], stats['acc_L2']['median'], stats['acc_L2']['q3'],
    #                                    stats['acc_L2']['max'], stats['acc_L2']['iqr'],
    #                                    stats['acc_L2']['skew'], stats['acc_L2']['kurtosis'],
    #                                    stats['acc_L2']['outlier_lower_3sigma'], stats['acc_L2']['outlier_upper_3sigma'],
    #                                    stats['acc_L2']['outlier_lower_quantile'],
    #                                    stats['acc_L2']['outlier_upper_quantile']]
    #
    #     des_stat.loc['acc_xy_L2'] = [stats['acc_xy_L2']['count'], stats['acc_xy_L2']['trimmed_mean'],
    #                                    stats['acc_xy_L2']['mean'],
    #                                    stats['acc_xy_L2']['std'], stats['acc_xy_L2']['min'],
    #                                    stats['acc_xy_L2']['q1'], stats['acc_xy_L2']['median'], stats['acc_xy_L2']['q3'],
    #                                    stats['acc_xy_L2']['max'], stats['acc_xy_L2']['iqr'],
    #                                    stats['acc_xy_L2']['skew'], stats['acc_xy_L2']['kurtosis'],
    #                                    stats['acc_xy_L2']['outlier_lower_3sigma'],
    #                                    stats['acc_xy_L2']['outlier_upper_3sigma'],
    #                                    stats['acc_xy_L2']['outlier_lower_quantile'],
    #                                    stats['acc_xy_L2']['outlier_upper_quantile']]
    #
    #     des_stat.loc['acc_z_abs'] = [stats['acc_z_abs']['count'], stats['acc_z_abs']['trimmed_mean'],
    #                                    stats['acc_z_abs']['mean'],
    #                                    stats['acc_z_abs']['std'], stats['acc_z_abs']['min'],
    #                                    stats['acc_z_abs']['q1'], stats['acc_z_abs']['median'], stats['acc_z_abs']['q3'],
    #                                    stats['acc_z_abs']['max'], stats['acc_z_abs']['iqr'],
    #                                    stats['acc_z_abs']['skew'], stats['acc_z_abs']['kurtosis'],
    #                                    stats['acc_z_abs']['outlier_lower_3sigma'],
    #                                    stats['acc_z_abs']['outlier_upper_3sigma'],
    #                                    stats['acc_z_abs']['outlier_lower_quantile'],
    #                                    stats['acc_z_abs']['outlier_upper_quantile']]
    #
    # des_stat.to_csv('../../dataset_distribution/original_data_distribution/all_data_descriptive_statistics.csv')

    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    print(f'所有文件处理完成！共使用{total_time / 60:.2f}分钟！')
