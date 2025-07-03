import glob
import os
import re
import time

import pandas as pd


def smooth_data(signal: pd.Series,
                    med_window: int = 5,
                    ema_span: int = 5,
                    med_center: bool = True) -> pd.Series:
    """
    对单通道序列做Median+EMA平滑，快速去噪并保留趋势。

    med_window: 中位数滤波窗口（推荐 3~7，必须为奇数）
    ema_span: EMA平滑窗口
    med_center: 为True则中位数滤波窗口两端对称，为False则只看之前的窗口
    """

    # 1. Median Filter 去除短时尖峰
    med_filtered = signal.rolling(
        window=med_window,
        center=med_center,      # med_center为True则窗口两端对称，为False则只看之前的窗口
        min_periods=1           # 窗口内至少 1 个非 NaN 就计算
    ).median()

    med_filtered = pd.Series(med_filtered, index=signal.index)

    # 2. EMA 平滑
    ema_smoothed = med_filtered.ewm(span=ema_span, adjust=False).mean()
    return ema_smoothed


if __name__ == '__main__':
    # 获取文件夹下所有 CSV 文件的路径
    # file_list = glob.glob('../../附件1/*.csv')
    file_list = glob.glob('../../附件2/*.csv')

    # 定义正则表达式
    pattern = re.compile(r'^(P\d+)\.csv$')

    # 过滤出符合条件的csv文件
    file_list = [f for f in file_list if pattern.match(os.path.basename(f))]
    print(f"共发现 {len(file_list)} 个符合的 CSV 文件")

    missing_df_all = pd.DataFrame()

    total_start_time = time.time()
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
        start_time = time.time()

        # 读取志愿者数据
        data_df = pd.read_csv(file_path, low_memory=False)

        # 从annotation中提取MET值
        # data_df['MET'] = data_df['annotation'].str.extract(r'MET ([\d\.]+)').astype(float)
        # data_df.drop(columns='annotation', inplace=True)

        data_df['x'] = smooth_data(data_df['x'], med_window=7, ema_span=100)
        data_df['y'] = smooth_data(data_df['y'], med_window=7, ema_span=100)
        data_df['z'] = smooth_data(data_df['z'], med_window=7, ema_span=100)
        # data_df['MET'] = smooth_data(data_df['MET'], med_window=7, ema_span=100)

        # data_df.to_csv('../../附件1/' + extracted_part + '.csv', index=False)
        data_df.to_csv('../../附件2/' + extracted_part + '.csv', index=False)

        end_time = time.time()
        one_file_time = end_time - start_time
        print(f'第{i + 1}个/共{len(file_list)}个，处理完成，用了{one_file_time / 60:.2f}分钟')

    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    print(f'数据平滑完成！共使用{total_time / 60:.2f}分钟！')
