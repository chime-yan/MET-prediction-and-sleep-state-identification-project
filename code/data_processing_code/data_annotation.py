import glob
import os
import re
import time

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt


def estimate_rem_confidence(sleep_time, total_sleep_time):
    """
    线性估计 REM 的置信度（用于 soft label）
    """
    if sleep_time < 5400:  # 刚入睡 90分钟 一般不会进入 REM
        return 0.1
    else:
        rel_progress = (sleep_time - 5400) / (total_sleep_time - 5400) \
            if (sleep_time > 5400) and (total_sleep_time > 5400) else 0

        return np.clip(0.1 + 0.8 * rel_progress, 0.1, 0.9)

def acc_l2_based_confidence(acc_l2, min_threshold, max_threshold):
    if acc_l2 < min_threshold:
        return 0.9
    else:
        # 越小越可能是REM → 映射到置信度 [0.1, 0.9]
        confidence = 0.9 - 0.8 * (
                (acc_l2 - min_threshold) / (max_threshold - min_threshold + 1e-8)
        )
    return np.clip(confidence, 0.1, 0.9)

def get_soft_label(df, min_threshold, max_threshold):
    df['time'] = pd.to_datetime(df['time'], format='mixed')
    df.sort_values('time', inplace=True)

    # 找出每段连续睡眠的起始点和结束点（sleep segment）
    df.loc['is_sleeping'] = 0
    df.loc[df['MET'] < 1, 'is_sleeping'] = 1

    df['is_sleeping'] = df['is_sleeping'].rolling(300, min_periods=1).mean()
    df['is_sleeping'] = (df['is_sleeping'] > 0.5).astype(int)

    df['sleep_segment'] = (df['is_sleeping'] != df['is_sleeping'].shift(1)).cumsum()

    # 删除因shift多出的最后一行
    df.drop('is_sleeping', axis=0, inplace=True)

    # 只保留睡眠段编号，非睡眠置为 NaN
    df.loc[df['is_sleeping'] == 0, 'sleep_segment'] = pd.NA

    df['sleep_time'] = 0
    df['rem_confidence'] = 0

    # 遍历每段睡眠
    for seg_id, seg_df in df.groupby('sleep_segment'):
        if pd.isna(seg_id):
            continue

        time_diff = df.loc[seg_df.index, 'time'].diff().fillna(pd.Timedelta(0))

        # 设置允许间隔
        allowed_diff = pd.Timedelta(seconds=1)

        # 过滤掉异常间隔，不是1秒的认为时间不连续，时间差设为0
        time_diff = (time_diff.where(time_diff == allowed_diff,
                                     pd.Timedelta(seconds=0)))

        # 计算累计时间（单位：秒）
        df.loc[seg_df.index, 'sleep_time'] = time_diff.dt.total_seconds().cumsum()

        df.loc[seg_df.index, 'rem_confidence'] = df.loc[seg_df.index, ['sleep_time', 'acc_L2']].apply(
            lambda x: (0.4 * estimate_rem_confidence(x['sleep_time'], x['sleep_time'].max())
                       + 0.6 * acc_l2_based_confidence(x['acc_L2'], min_threshold, max_threshold)),
            axis=1
        )

    # 数据标注
    df['state_NO_SLEEP'] = df['is_sleeping'].transform(lambda x: 0 if x == 1 else 1)
    df['state_NREM'] = 0
    df['state_REM'] = 0

    df.loc[df['state_NO_SLEEP'] == 0, 'state_NREM'] = 1 - df['rem_confidence']
    df.loc[df['state_NO_SLEEP'] == 0, 'state_REM'] = df['rem_confidence']

    df.drop(['is_sleeping', 'sleep_segment', 'sleep_time', 'rem_confidence'],
            axis=1, inplace=True)


if __name__ == '__main__':
    # 获取文件夹下所有 CSV 文件的路径
    file_list = glob.glob('../../附件1/*.csv')

    # 定义正则表达式
    pattern = re.compile(r'^(P\d+)\.csv$')

    # 过滤出符合条件的csv文件
    file_list = [f for f in file_list if pattern.match(os.path.basename(f))]
    print(f"共发现 {len(file_list)} 个符合的 CSV 文件")

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

        print(f'第{i + 1}个/共{len(file_list)}个，开始标注{extracted_part}')
        start_time = time.time()

        # 读取志愿者数据
        data_df = pd.read_csv(file_path)
        data_df.sort_values(by="time", inplace=True, ascending=True)

        # 计算 MET<1 时的 REM 加速度阈值
        max_threshold = data_df.loc[data_df['MET'] < 1, 'acc_L2'].quantile(0.25)
        min_threshold = data_df.loc[data_df['MET'] < 1, 'acc_L2'].quantile(0.1)

        # 数据标注（软标签），state_NO_SLEEP 为未睡眠状态，state_NREM 为 NREM 状态，state_REM 为 REM 状态
        get_soft_label(data_df, min_threshold, max_threshold)

        # 由 train_data_descriptive_statistics.csv 可知，acc_L2 的标准差太低(只有约0.054)，故将 MET 值与之相乘，构造新的特征 accL2_MET
        data_df['accL2_MET'] = data_df['acc_L2'] * data_df['MET']

        data_df.to_csv('../../附件1/' + extracted_part + '.csv', index=False)

        end_time = time.time()
        one_file_time = end_time - start_time
        print(f'标注{extracted_part}文件的睡眠状态完成，用了{one_file_time / 60:.2f}分钟')


        # plt.figure(figsize=(16, 8))
        # plt.scatter(data_df.index, data_df['state_NO_SLEEP'], alpha=0.5, s=1)
        # plt.xticks(range(0, len(data_df.index), 1800),
        #            labels=[f'{i/3600: .1f}h' for i in range(0, len(data_df.index), 1800)],
        #            rotation=45)
        #
        # plt.ylim(-1, 4)
        # plt.yticks(range(-1, 4))
        # plt.xlabel('Cumulative Time')
        # plt.ylabel('is_NO_SLEEP')
        # plt.title(f'{extracted_part} is_NO_SLEEP Scatter')
        #
        # os.makedirs('../../睡眠状态随时间变化的散点图', exist_ok=True)
        # plt.savefig('../../睡眠状态随时间变化的散点图/' + extracted_part + '_sleep_state_scatter.png')
        # plt.close()

    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    print(f'所有文件完成睡眠状态标注！共使用{total_time / 60:.2f}分钟！')
