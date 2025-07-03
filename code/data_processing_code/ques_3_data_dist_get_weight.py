import numpy as np
import pandas as pd
from tdigest import TDigest

from data_dist_estimation import WelfordOnlineStats


def compute_effective_counts(targets, certain_class_idx=0, uncertainty_threshold=0.8):
    """
    只计算 effective_counts
    """

    num_classes = targets.shape[1]
    effective_counts = np.zeros(num_classes)

    for class_idx in range(num_classes):
        if class_idx == certain_class_idx:
            effective_counts[class_idx] = (targets[:, class_idx] > uncertainty_threshold).sum()
        else:
            class_probs = targets[:, class_idx]
            sample_entropy = -(targets * np.log(targets + 1e-8)).sum(axis=1)
            max_entropy = np.log(num_classes)
            confidence = 1.0 - (sample_entropy / max_entropy)
            effective_counts[class_idx] = (class_probs * confidence).sum()

    return effective_counts


if __name__ == '__main__':
    dataset_path_csv_path = '../../output_and_result_files/dataset_paths.csv'

    # 获取训练数据的所有 CSV 文件的路径
    dataset_path = pd.read_csv(dataset_path_csv_path)
    train_file_list = list(dataset_path.loc[dataset_path['type'] == 'train', 'file_path'])

    class_dict = {0: 'NO_SLEEP', 1: 'NREM', 2: 'REM'}

    # 初始化类别计数字典
    class_counts = {}
    # 假设已知类别为0, 1, 2
    for label in [0, 1, 2]:
        class_counts[label] = 0

    des_est_df = pd.read_csv(
        '../../dataset_distribution/original_data_distribution/train_data_descriptive_statistics.csv', index_col=0)

    # 维护 TDigest 实例和 WelfordOnlineStats 实例
    td2 = TDigest()
    wost2 = WelfordOnlineStats()

    print(f"开始收集归一化数据和进行类别计数！")
    for i, file_path in enumerate(train_file_list):
        print(f'开始遍历第{i + 1}个训练集中的csv文件，共{len(train_file_list)}个')

        # 读取训练集中志愿者数据
        data_df = pd.read_csv(file_path, usecols=['accL2_MET',
                                                  'state_NO_SLEEP', 'state_NREM', 'state_REM'])

        values2 = data_df['accL2_MET'].values

        for v2 in values2:
            td2.update(v2)
            wost2.update(v2)

        # 类别计数
        class_prob = data_df[['state_NO_SLEEP', 'state_NREM', 'state_REM']].values
        effect_class_counts = compute_effective_counts(class_prob, certain_class_idx=0)
        for label in range(class_prob.shape[1]):
            class_counts[label] += effect_class_counts[label]

    median_2 = td2.percentile(50)
    q1_2 = td2.percentile(25)
    q3_2 = td2.percentile(75)
    iqr_2 = q3_2 - q1_2

    # 计算修剪均值，去除最低 5% 和最高 5% 的数据
    trimmed_mean_2 = td2.trimmed_mean(5, 95)

    wost_kv_2 = wost2.finalize()

    outlier_lower_3sigma_2 = wost_kv_2['mean'] - 3 * wost_kv_2['std']
    outlier_upper_3sigma_2 = wost_kv_2['mean'] + 3 * wost_kv_2['std']
    outlier_lower_quantile_2 = q1_2 - 1.5 * iqr_2
    outlier_upper_quantile_2 = q3_2 + 1.5 * iqr_2

    if 'accL2_MET' not in des_est_df.index:
        des_est_df.loc[len(des_est_df)] = [wost_kv_2['count'], trimmed_mean_2, wost_kv_2['mean'],
                                           wost_kv_2['std'], wost_kv_2['min'], q1_2, median_2, q3_2,
                                           wost_kv_2['max'], iqr_2, wost_kv_2['skew'], wost_kv_2['kurtosis'],
                                           outlier_lower_3sigma_2, outlier_upper_3sigma_2,
                                           outlier_lower_quantile_2, outlier_upper_quantile_2]
        des_est_df.index = ['x', 'y', 'z', 'MET', 'acc_L2', 'acc_xy_L2', 'acc_z_abs', 'boxcox_acc_L2', 'accL2_MET']

    else:
        des_est_df.loc['accL2_MET'] = [wost_kv_2['count'], trimmed_mean_2, wost_kv_2['mean'],
                                           wost_kv_2['std'], wost_kv_2['min'], q1_2, median_2, q3_2,
                                           wost_kv_2['max'], iqr_2, wost_kv_2['skew'], wost_kv_2['kurtosis'],
                                           outlier_lower_3sigma_2, outlier_upper_3sigma_2,
                                           outlier_lower_quantile_2, outlier_upper_quantile_2]

    des_est_df.to_csv('../../dataset_distribution/original_data_distribution/train_data_descriptive_statistics.csv')
    print(f"收集归一化数据和类别计数完成！")


    # 计算每个类别的频率的倒数
    class_weights = {label: 1.0 / (count + 1e-8) for label, count in class_counts.items()}

    # 归一化权重，使得权重总和为3
    total_weight = sum(class_weights.values())
    norm_class_weights = {label: weight / total_weight * len(class_weights) for label, weight in class_weights.items()}

    # 将权重转换为DataFrame
    df_class_weights = pd.DataFrame(list(class_weights.items()), columns=['label', 'weight'])
    df_norm_class_weights = pd.DataFrame(list(norm_class_weights.items()), columns=['label', 'norm_weight'])

    # 合并两个DataFrame
    df = pd.merge(df_class_weights, df_norm_class_weights, on='label')
    df['class'] = df['label'].map(class_dict)

    df.to_csv('../../output_and_result_files/ques_3_category_weights.csv', index=False)
    print(f"类别权重已保存到 ../../output_and_result_files/ques_3_category_weights.csv ")
