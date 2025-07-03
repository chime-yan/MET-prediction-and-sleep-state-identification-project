import glob
import os
import re
import time

import pandas as pd


# 定义处理加速度的函数
def process_acceleration(acceleration_data):
    # 计算绝对值的平均值
    abs_mean = acceleration_data.abs().mean()
    # 计算原始数据的平均值
    mean = acceleration_data.mean()
    # 合并特征：将方向符号应用到绝对值平均值上
    signed_abs_mean = abs_mean * (1 if mean >= 0 else -1)
    return signed_abs_mean


if __name__ == '__main__':
    # 获取文件夹下所有 CSV 文件的路径
    # file_list = glob.glob('../../附件1/*.csv')
    file_list = glob.glob('../../附件2/*.csv')

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

        print(f'第{i + 1}个/共{len(file_list)}个，开始处理{extracted_part}')
        start_time = time.time()

        # 读取志愿者数据
        data_df = pd.read_csv(file_path, low_memory=False)

        data_df['time'] = pd.to_datetime(data_df['time'], format='mixed')
        data_df.sort_values('time', inplace=True)

        # 设置时间索引
        data_df.set_index('time', inplace=True)

        # 按1秒窗口分组并聚合
        # 取第一个值
        # 按1秒窗口分组并处理加速度数据
        # data_df = data_df.resample('1s').agg({
        #     'x': process_acceleration,
        #     'y': process_acceleration,
        #     'z': process_acceleration,
        #     'MET': 'mean'
        # })

        data_df = data_df.resample('1s').agg({
            'x': process_acceleration,
            'y': process_acceleration,
            'z': process_acceleration
        })

        # 删除时间戳可能的中断部分数据
        # data_df.dropna(subset=['x', 'y', 'z', 'MET'], how='all', inplace=True)
        data_df.dropna(subset=['x', 'y', 'z'], how='all', inplace=True)

        # 重置索引，将时间列恢复为普通列
        data_df.reset_index(inplace=True)

        # data_df.to_csv('../../附件1/' + extracted_part + '.csv', index=False)
        data_df.to_csv('../../附件2/' + extracted_part + '.csv', index=False)

        end_time = time.time()
        one_file_time = end_time - start_time
        print(f'第{i + 1}个/共{len(file_list)}个，处理完成，用了{one_file_time / 60:.2f}分钟')

    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    print(f'降采样完成！共使用{total_time / 60:.2f}分钟！')
