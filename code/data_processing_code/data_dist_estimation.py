import pandas as pd
from tdigest import TDigest
import math


def get_statistics_for_dict(statis_df, feature: str, stats: dict):
    statis_df.loc[feature, 'median'] = stats[feature]['median']
    statis_df.loc[feature, 'iqr'] = stats[feature]['iqr']
    statis_df.loc[feature, 'trimmed_mean'] = stats[feature]['trimmed_mean']
    statis_df.loc[feature, 'q1'] = stats[feature]['q1']
    statis_df.loc[feature, 'q3'] = stats[feature]['q3']

    statis_df.loc[feature, 'count'] = stats[feature]['count']
    statis_df.loc[feature, 'mean'] = stats[feature]['mean']
    statis_df.loc[feature, 'std'] = stats[feature]['std']
    statis_df.loc[feature, 'min'] = stats[feature]['min']
    statis_df.loc[feature, 'max'] = stats[feature]['max']

    statis_df.loc[feature, 'skew'] = stats[feature]['skew']
    statis_df.loc[feature, 'kurtosis'] = stats[feature]['kurtosis']

    statis_df.loc[feature, 'outlier_lower_3sigma'] = stats[feature]['outlier_lower_3sigma']
    statis_df.loc[feature, 'outlier_upper_3sigma'] = stats[feature]['outlier_upper_3sigma']
    statis_df.loc[feature, 'outlier_lower_quantile'] = stats[feature]['outlier_lower_quantile']
    statis_df.loc[feature, 'outlier_upper_quantile'] = stats[feature]['outlier_upper_quantile']


class WelfordOnlineStats:
    def __init__(self):
        self.n = 0          # 样本数量
        self.mean = 0.0     # 均值
        self.M2 = 0.0       # 用于标准差计算的累计量（累积二阶矩）
        self.M3 = 0.0       # 累积三阶矩
        self.M4 = 0.0       # 累积四阶矩

        self.min_val = float("inf")
        self.max_val = float("-inf")

    def update(self, x):
        self.n += 1

        delta = x - self.mean
        delta_n = delta / self.n
        delta_n2 = delta_n * delta_n
        term1 = delta * delta_n * (self.n - 1)

        self.mean += delta_n
        self.M4 += term1 * delta_n2 * (self.n * self.n - 3 * self.n + 3) + 6 * delta_n2 * self.M2 - 4 * delta_n * self.M3
        self.M3 += term1 * delta_n * (self.n - 2) - 3 * delta_n * self.M2
        self.M2 += term1

        self.min_val = min(self.min_val, x)
        self.max_val = max(self.max_val, x)

    def finalize(self):
        variance = self.M2 / (self.n - 1) if self.n > 1 else float('nan')
        skewness = ((math.sqrt(self.n * (self.n - 1)) / (self.n - 2)) * (self.M3 / (self.M2 ** 1.5))) \
            if (self.n > 2) and (self.M2 != 0) else float('nan')

        kurtosis = ((self.n - 1) * ((self.n + 1) * (self.n * self.M4) / (self.M2 * self.M2) - 3 * (self.n - 1)) / ((self.n - 2) * (self.n - 3))) \
            if (self.n > 3) and (self.M2 != 0) else float('nan')

        stddev = math.sqrt(variance)

        return {
            'count': self.n,
            'mean': self.mean,
            'std': stddev,
            'min': self.min_val,
            'max': self.max_val,
            'skew': skewness,
            'kurtosis': kurtosis
        }


def merge_nested_dicts(d1, d2):
    merged = {}
    for key in d1.keys() | d2.keys():
        if key in d1 and key in d2:
            # 如果两个值都是字典，递归合并
            if isinstance(d1[key], dict) and isinstance(d2[key], dict):
                merged[key] = merge_nested_dicts(d1[key], d2[key])
            else:
                # 如果值不是字典，d2 的值覆盖 d1
                merged[key] = d2[key]
        elif key in d1:
            merged[key] = d1[key]
        else:
            merged[key] = d2[key]
    return merged

def collect_statistics(target_columns : list, file_list : list):
    # 为每一列维护一个 TDigest 实例和 WelfordOnlineStats 实例
    td_dict = {col: TDigest() for col in target_columns}
    wost_dict = {col: WelfordOnlineStats() for col in target_columns}

    print(f"开始收集统计数据！")
    for i, file_path in enumerate(file_list):
        print(f'开始遍历第{i+1}个csv文件，共{len(file_list)}个')

        # 读取志愿者数据
        data_df = pd.read_csv(file_path, usecols=target_columns)

        for col in target_columns:
            values = data_df[col].dropna().values
            for v in values:
                td_dict[col].update(v)
                wost_dict[col].update(v)

    # 计算每一列的修剪均值、全局中位数、IQR、样本总数、最大值、最小值、全局均值和标准差
    statis_dict = {}
    statis_dict_2 = {}
    statis_dict_3 = {}
    for col in target_columns:
        td = td_dict[col]
        wost = wost_dict[col]

        median = td.percentile(50)
        q1 = td.percentile(25)
        q3 = td.percentile(75)
        iqr = q3 - q1

        # 计算修剪均值，去除最低 5% 和最高 5% 的数据
        trimmed_mean = td.trimmed_mean(5, 95)

        statis_dict[col] = {
            "trimmed_mean": trimmed_mean,
            "median": median,
            "q1": q1,
            "q3": q3,
            "iqr": iqr
        }

        statis_dict_2[col] = wost.finalize()

        statis_dict_3[col] = {
            "outlier_lower_3sigma": statis_dict_2[col]["mean"] - 3 * statis_dict_2[col]["std"],
            "outlier_upper_3sigma": statis_dict_2[col]["mean"] + 3 * statis_dict_2[col]["std"],

            "outlier_lower_quantile": statis_dict[col]["q1"] - 1.5 * statis_dict[col]["iqr"],
            "outlier_upper_quantile": statis_dict[col]["q3"] + 1.5 * statis_dict[col]["iqr"]
        }

    statis_dict = merge_nested_dicts(statis_dict, statis_dict_2)
    statis_dict = merge_nested_dicts(statis_dict, statis_dict_3)
    return statis_dict


if __name__ == '__main__':
    dataset_path_csv_path = '../../output_and_result_files/dataset_paths.csv'
    dataset_path = pd.read_csv(dataset_path_csv_path)

    # 获取训练数据的所有 CSV 文件的路径
    # file_list = list(dataset_path.loc[dataset_path['type'] == 'train', 'file_path'])

    # 获取验证数据的所有 CSV 文件的路径
    # file_list = list(dataset_path.loc[dataset_path['type'] == 'val', 'file_path'])

    # 获取测试数据的所有 CSV 文件的路径
    file_list = list(dataset_path.loc[dataset_path['type'] == 'test', 'file_path'])

    target_columns = ['x', 'y', 'z', 'MET', 'acc_L2', 'acc_xy_L2', 'acc_z_abs']
    des_stat = pd.DataFrame(index=['x', 'y', 'z', 'MET', 'acc_L2', 'acc_xy_L2', 'acc_z_abs'],
                            columns=['count', 'trimmed_mean', 'mean', 'std',
                                    'min', 'q1', 'median', 'q3', 'max', 'iqr',
                                     'skew', 'kurtosis',
                                     'outlier_lower_3sigma', 'outlier_upper_3sigma',
                                     'outlier_lower_quantile', 'outlier_upper_quantile'
                                     ])

    stats = collect_statistics(target_columns, file_list)

    get_statistics_for_dict(des_stat, 'x', stats)
    get_statistics_for_dict(des_stat, 'y', stats)
    get_statistics_for_dict(des_stat, 'z', stats)
    get_statistics_for_dict(des_stat, 'MET', stats)
    get_statistics_for_dict(des_stat, 'acc_L2', stats)
    get_statistics_for_dict(des_stat, 'acc_xy_L2', stats)
    get_statistics_for_dict(des_stat, 'acc_z_abs', stats)

    # des_stat.to_csv('../../dataset_distribution/original_data_distribution/train_data_descriptive_statistics.csv')
    # des_stat.to_csv('../../dataset_distribution/original_data_distribution/val_data_descriptive_statistics.csv')
    des_stat.to_csv('../../dataset_distribution/original_data_distribution/test_data_descriptive_statistics.csv')

    print(f"收集统计数据完成！")
