import os
import re
import time

import pandas as pd
import glob


def data_statistics(data_df):
    data_df['time'] = pd.to_datetime(data_df['time'], format='mixed')

    # 计算睡眠时间
    time_df = data_df.loc[data_df['MET'] < 1, ['time']].sort_values(by=['time'])

    # 计算每个时间与前一个时间的差值（时间间隔）
    time_df['time_diff'] = time_df['time'].diff()

    # 过滤掉异常时间差（例如，超过1秒的时间差）
    max_allowed_diff = pd.Timedelta(seconds=1)
    valid_diffs = time_df['time_diff'][time_df['time_diff'] <= max_allowed_diff]

    # 计算所有时间差的总和
    total_time_diff = valid_diffs.sum()
    sleep_time = round(total_time_diff.total_seconds() / 60 / 60, 4)


    # 计算高等强度运动时间
    time_df = data_df.loc[data_df['MET'] >= 6, ['time']].sort_values(by=['time'])

    time_df['time_diff'] = time_df['time'].diff()

    # 过滤掉异常时间差（例如，超过1秒的时间差）
    max_allowed_diff = pd.Timedelta(seconds=1)
    valid_diffs = time_df['time_diff'][time_df['time_diff'] <= max_allowed_diff]

    total_time_diff = valid_diffs.sum()
    high_time = round(total_time_diff.total_seconds() / 60 / 60, 4)


    # 计算中等强度运动时间
    time_df = data_df.loc[((data_df['MET'] >= 3) & (data_df['MET'] < 6)), ['time']].sort_values(by=['time'])

    time_df['time_diff'] = time_df['time'].diff()

    # 过滤掉异常时间差（例如，超过1秒的时间差）
    max_allowed_diff = pd.Timedelta(seconds=1)
    valid_diffs = time_df['time_diff'][time_df['time_diff'] <= max_allowed_diff]

    total_time_diff = valid_diffs.sum()
    medium_time = round(total_time_diff.total_seconds() / 60 / 60, 4)


    # 计算低等强度运动时间
    time_df = data_df.loc[((data_df['MET'] >= 1.6) & (data_df['MET'] < 3)), ['time']].sort_values(by=['time'])

    time_df['time_diff'] = time_df['time'].diff()

    # 过滤掉异常时间差（例如，超过1秒的时间差）
    max_allowed_diff = pd.Timedelta(seconds=1)
    valid_diffs = time_df['time_diff'][time_df['time_diff'] <= max_allowed_diff]

    total_time_diff = valid_diffs.sum()
    small_time = round(total_time_diff.total_seconds() / 60 / 60, 4)


    # 计算静态活动总时长
    time_df = data_df.loc[(data_df['MET'] >= 1) & (data_df['MET'] < 1.6), ['time']].sort_values(by=['time'])

    time_df['time_diff'] = time_df['time'].diff()

    # 过滤掉异常时间差（例如，超过1秒的时间差）
    max_allowed_diff = pd.Timedelta(seconds=1)
    valid_diffs = time_df['time_diff'][time_df['time_diff'] <= max_allowed_diff]

    total_time_diff = valid_diffs.sum()
    static_time = round(total_time_diff.total_seconds() / 60 / 60, 4)


    # 计算记录总时长（小时）
    overall_time = round(sleep_time + high_time + medium_time + small_time + static_time, 4)
    return sleep_time, high_time, medium_time, small_time, static_time, overall_time


if __name__ == '__main__':
    # 获取文件夹下所有 CSV 文件的路径
    file_list = glob.glob('../../附件1/*.csv')

    # 定义正则表达式
    pattern = re.compile(r'^(P\d+)\.csv$')

    # 过滤出符合条件的csv文件
    file_list = [f for f in file_list if pattern.match(os.path.basename(f))]
    print(f"共发现 {len(file_list)} 个符合的 CSV 文件")

    df_all = pd.DataFrame()
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
        data_df = pd.read_csv(file_path, usecols=['time', 'MET'])

        sleep_time, high_time, medium_time, small_time, static_time, overall_time = data_statistics(data_df)
        df = pd.DataFrame({'志愿者ID': [extracted_part],
                           '记录总时长（小时）': [f'{overall_time:.4f}'],
                           '睡眠总时长（小时）': [f'{sleep_time:.4f}'],
                           '高等强度运动总时长（小时）': [f'{high_time:.4f}'],
                           '中等强度运动总时长（小时）': [f'{medium_time:.4f}'],
                           '低等强度运动总时长（小时）': [f'{small_time:.4f}'],
                           '静态活动总时长（小时）': [f'{static_time:.4f}']})

        # 追加到总 DataFrame
        df_all = pd.concat([df_all, df], ignore_index=True)

        end_time = time.time()
        one_file_time = end_time - start_time
        print(f'统计{extracted_part}文件的时长数据完成，用了{one_file_time / 60:.2f}分钟')

    df_all.to_excel('../../output_and_result_files/result_1.xlsx', index=False)
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    print(f'统计所有文件的时长数据完成！共使用{total_time / 60:.2f}分钟！')
