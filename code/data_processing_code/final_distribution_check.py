import os
import pandas as pd

from data_dist_estimation import collect_statistics, get_statistics_for_dict


if __name__ == '__main__':
    dataset_path_csv_path = '../../output_and_result_files/dataset_paths.csv'
    dataset_path = pd.read_csv(dataset_path_csv_path)

    # 获取训练数据的所有 CSV 文件的路径
    file_list = list(dataset_path.loc[dataset_path['type'] == 'train', 'file_path'])

    # 获取验证数据的所有 CSV 文件的路径
    # file_list = list(dataset_path.loc[dataset_path['type'] == 'val', 'file_path'])

    # 获取测试数据的所有 CSV 文件的路径
    # file_list = list(dataset_path.loc[dataset_path['type'] == 'test', 'file_path'])

    target_columns = ['norm_x', 'norm_y', 'norm_z', 'norm_boxcox_acc_L2', 'norm_acc_xy_L2', 'norm_acc_z_abs',
                      'norm_MET_target', 'norm_MET', 'norm_accL2_MET']
    des_stat = pd.DataFrame(index=['norm_x', 'norm_y', 'norm_z', 'norm_boxcox_acc_L2', 'norm_acc_xy_L2', 'norm_acc_z_abs',
                      'norm_MET_target', 'norm_MET', 'norm_accL2_MET'],
                            columns=['count', 'trimmed_mean', 'mean', 'std',
                                    'min', 'q1', 'median', 'q3', 'max', 'iqr',
                                     'skew', 'kurtosis',
                                     'outlier_lower_3sigma', 'outlier_upper_3sigma',
                                     'outlier_lower_quantile', 'outlier_upper_quantile'
                                     ])

    stats = collect_statistics(target_columns, file_list)

    get_statistics_for_dict(des_stat, 'norm_x', stats)
    get_statistics_for_dict(des_stat, 'norm_y', stats)
    get_statistics_for_dict(des_stat, 'norm_z', stats)
    get_statistics_for_dict(des_stat, 'norm_boxcox_acc_L2', stats)
    get_statistics_for_dict(des_stat, 'norm_acc_xy_L2', stats)
    get_statistics_for_dict(des_stat, 'norm_acc_z_abs', stats)
    get_statistics_for_dict(des_stat, 'norm_MET_target', stats)
    get_statistics_for_dict(des_stat, 'norm_MET', stats)
    get_statistics_for_dict(des_stat, 'norm_accL2_MET', stats)

    os.makedirs('../../dataset_distribution/norm_data_distribution', exist_ok=True)
    des_stat.to_csv('../../dataset_distribution/norm_data_distribution/train_used_dist.csv')
    # des_stat.to_csv('../../dataset_distribution/norm_data_distribution/val_used_dist.csv')
    # des_stat.to_csv('../../dataset_distribution/norm_data_distribution/test_used_dist.csv')

    print(f"收集用于训练、验证和测试数据的统计数据完成！")
