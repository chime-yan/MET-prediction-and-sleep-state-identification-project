import glob
import os
import re

import pandas as pd

from data_dist_estimation import collect_statistics


if __name__ == '__main__':
    # 获取文件夹下所有 CSV 文件的路径
    file_list = glob.glob('../../附件1/*.csv')

    # 定义正则表达式
    pattern = re.compile(r'^(P\d+)\.csv$')

    # 过滤出符合条件的csv文件
    file_list = [f for f in file_list if pattern.match(os.path.basename(f))]
    print(f"共发现 {len(file_list)} 个符合的 CSV 文件")

    dtypes_df_all = pd.DataFrame()
    missing_df_all = pd.DataFrame()
    abnormal_df_all = pd.DataFrame()
    repeat_df_all = pd.DataFrame()

    stats = collect_statistics(['x', 'y', 'z', 'MET'], file_list)
    des_stat = pd.DataFrame(index=['x', 'y', 'z', 'MET'],
                            columns=['count', 'trimmed_mean', 'mean', 'std',
                                     'min', 'q1', 'median', 'q3', 'max', 'iqr',
                                     'skew', 'kurtosis',
                                     'outlier_lower_3sigma', 'outlier_upper_3sigma',
                                     'outlier_lower_quantile', 'outlier_upper_quantile'])


    des_stat.loc['x', 'median'] = stats['x']['median']
    des_stat.loc['x', 'iqr'] = stats['x']['iqr']
    des_stat.loc['x', 'trimmed_mean'] = stats['x']['trimmed_mean']
    des_stat.loc['x', 'q1'] = stats['x']['q1']
    des_stat.loc['x', 'q3'] = stats['x']['q3']

    des_stat.loc['x', 'count'] = stats['x']['count']
    des_stat.loc['x', 'mean'] = stats['x']['mean']
    des_stat.loc['x', 'std'] = stats['x']['std']
    des_stat.loc['x', 'min'] = stats['x']['min']
    des_stat.loc['x', 'max'] = stats['x']['max']

    des_stat.loc['x', 'skew'] = stats['x']['skew']
    des_stat.loc['x', 'kurtosis'] = stats['x']['kurtosis']

    des_stat.loc['x', 'outlier_lower_3sigma'] = stats['x']['outlier_lower_3sigma']
    des_stat.loc['x', 'outlier_upper_3sigma'] = stats['x']['outlier_upper_3sigma']
    des_stat.loc['x', 'outlier_lower_quantile'] = stats['x']['outlier_lower_quantile']
    des_stat.loc['x', 'outlier_upper_quantile'] = stats['x']['outlier_upper_quantile']


    des_stat.loc['y', 'median'] = stats['y']['median']
    des_stat.loc['y', 'iqr'] = stats['y']['iqr']
    des_stat.loc['y', 'trimmed_mean'] = stats['y']['trimmed_mean']
    des_stat.loc['y', 'q1'] = stats['y']['q1']
    des_stat.loc['y', 'q3'] = stats['y']['q3']

    des_stat.loc['y', 'count'] = stats['y']['count']
    des_stat.loc['y', 'mean'] = stats['y']['mean']
    des_stat.loc['y', 'std'] = stats['y']['std']
    des_stat.loc['y', 'min'] = stats['y']['min']
    des_stat.loc['y', 'max'] = stats['y']['max']

    des_stat.loc['y', 'skew'] = stats['y']['skew']
    des_stat.loc['y', 'kurtosis'] = stats['y']['kurtosis']

    des_stat.loc['y', 'outlier_lower_3sigma'] = stats['y']['outlier_lower_3sigma']
    des_stat.loc['y', 'outlier_upper_3sigma'] = stats['y']['outlier_upper_3sigma']
    des_stat.loc['y', 'outlier_lower_quantile'] = stats['y']['outlier_lower_quantile']
    des_stat.loc['y', 'outlier_upper_quantile'] = stats['y']['outlier_upper_quantile']


    des_stat.loc['z', 'median'] = stats['z']['median']
    des_stat.loc['z', 'iqr'] = stats['z']['iqr']
    des_stat.loc['z', 'trimmed_mean'] = stats['z']['trimmed_mean']
    des_stat.loc['z', 'q1'] = stats['z']['q1']
    des_stat.loc['z', 'q3'] = stats['z']['q3']

    des_stat.loc['z', 'count'] = stats['z']['count']
    des_stat.loc['z', 'mean'] = stats['z']['mean']
    des_stat.loc['z', 'std'] = stats['z']['std']
    des_stat.loc['z', 'min'] = stats['z']['min']
    des_stat.loc['z', 'max'] = stats['z']['max']

    des_stat.loc['z', 'skew'] = stats['z']['skew']
    des_stat.loc['z', 'kurtosis'] = stats['z']['kurtosis']

    des_stat.loc['z', 'outlier_lower_3sigma'] = stats['z']['outlier_lower_3sigma']
    des_stat.loc['z', 'outlier_upper_3sigma'] = stats['z']['outlier_upper_3sigma']
    des_stat.loc['z', 'outlier_lower_quantile'] = stats['z']['outlier_lower_quantile']
    des_stat.loc['z', 'outlier_upper_quantile'] = stats['z']['outlier_upper_quantile']


    des_stat.loc['MET', 'median'] = stats['MET']['median']
    des_stat.loc['MET', 'iqr'] = stats['MET']['iqr']
    des_stat.loc['MET', 'trimmed_mean'] = stats['MET']['trimmed_mean']
    des_stat.loc['MET', 'q1'] = stats['MET']['q1']
    des_stat.loc['MET', 'q3'] = stats['MET']['q3']

    des_stat.loc['MET', 'count'] = stats['MET']['count']
    des_stat.loc['MET', 'mean'] = stats['MET']['mean']
    des_stat.loc['MET', 'std'] = stats['MET']['std']
    des_stat.loc['MET', 'min'] = stats['MET']['min']
    des_stat.loc['MET', 'max'] = stats['MET']['max']

    des_stat.loc['MET', 'skew'] = stats['MET']['skew']
    des_stat.loc['MET', 'kurtosis'] = stats['MET']['kurtosis']

    des_stat.loc['MET', 'outlier_lower_3sigma'] = stats['MET']['outlier_lower_3sigma']
    des_stat.loc['MET', 'outlier_upper_3sigma'] = stats['MET']['outlier_upper_3sigma']
    des_stat.loc['MET', 'outlier_lower_quantile'] = stats['MET']['outlier_lower_quantile']
    des_stat.loc['MET', 'outlier_upper_quantile'] = stats['MET']['outlier_upper_quantile']

    des_stat.to_csv('../../dataset_distribution/original_data_distribution/all_data_descriptive_statistics.csv')

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

        # 数据概览
        # 数据特征类型
        dtypes_df = pd.DataFrame({
            'pid': [extracted_part],
            'time': [data_df['time'].dtypes],
            'x': [data_df['x'].dtypes],
            'y': [data_df['y'].dtypes],
            'z': [data_df['z'].dtypes]
        })

        # 数据缺失检测
        overall_count = data_df.shape[0]
        x_missing_count = data_df['x'].isnull().sum()
        y_missing_count = data_df['y'].isnull().sum()
        z_missing_count = data_df['z'].isnull().sum()
        met_missing_count = data_df['MET'].isnull().sum()

        missing_df = pd.DataFrame({
            'pid': [extracted_part],
            'time': [data_df['time'].isnull().sum()],
            'x': [x_missing_count],
            'y': [y_missing_count],
            'z': [z_missing_count],
            'MET': [met_missing_count],
            '所有数据的总数据量': [overall_count],
            'x缺失数据占比': [round(x_missing_count / overall_count, 4)],
            'y缺失数据占比': [round(y_missing_count / overall_count, 4)],
            'z缺失数据占比': [round(z_missing_count / overall_count, 4)],
            'MET缺失数据占比': [round(met_missing_count / overall_count, 4)]
        })

        # 记录重复检测
        repeat_df = pd.DataFrame({
            'pid': [extracted_part],
            '重复的记录': [data_df.duplicated().sum()]
        })

        dtypes_df_all = pd.concat([dtypes_df_all, dtypes_df], ignore_index=True)
        missing_df_all = pd.concat([missing_df_all, missing_df], ignore_index=True)
        repeat_df_all = pd.concat([repeat_df_all, repeat_df], ignore_index=True)

    os.makedirs('../../data_glance_results', exist_ok=True)
    dtypes_df_all.to_excel('../../data_glance_results/data_dtypes.xlsx', index=False)
    missing_df_all.to_excel('../../data_glance_results/data_missing.xlsx', index=False)
    repeat_df_all.to_excel('../../data_glance_results/data_repeat.xlsx', index=False)
    print('数据初步概览完成！')
