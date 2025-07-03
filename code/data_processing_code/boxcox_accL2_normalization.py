import os
import re
import time

import pandas as pd

from data_normalization_and_encoding import normalization


if __name__ == '__main__':
    dataset_path_csv_path = '../../output_and_result_files/dataset_paths.csv'

    # 获取训练数据的所有 CSV 文件的路径
    dataset_path = pd.read_csv(dataset_path_csv_path)
    train_file_list = list(dataset_path.loc[dataset_path['type'] == 'train', 'file_path'])

    des_est_df = pd.read_csv(
        '../../dataset_distribution/original_data_distribution/train_data_descriptive_statistics.csv', index_col=0)
    boxcox_acc_l2_median = des_est_df.loc['boxcox_acc_L2', 'median']
    boxcox_acc_l2_iqr = des_est_df.loc['boxcox_acc_L2', 'iqr']

    # 定义正则表达式
    pattern = re.compile(r'^(P\d+)\.csv$')

    # 获取验证数据的所有 CSV 文件的路径
    val_file_list = list(dataset_path.loc[dataset_path['type'] == 'val', 'file_path'])

    # 获取测试数据的所有 CSV 文件的路径
    test_file_list = list(dataset_path.loc[dataset_path['type'] == 'test', 'file_path'])

    file_list_list = [train_file_list, val_file_list, test_file_list]

    print(f"开始进行归一化")
    total_start_time = time.time()


    print('------------------处理变换后的 boxcox_acc_L2 ---------------------')
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

            # Robust Z-score 归一化
            data_df['norm_boxcox_acc_L2'] = normalization(data_df['boxcox_acc_L2'], boxcox_acc_l2_median,
                                                          boxcox_acc_l2_iqr, alpha=1.0/3.0)

            data_df.to_csv('../../附件1/' + extracted_part + '.csv', index=False)

            end_time = time.time()
            one_file_time = end_time - start_time
            print(f'处理{extracted_part}文件用了{one_file_time / 60:.2f}分钟')

    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    print(f'数据归一化完成！共使用{total_time / 60:.2f}分钟！')
