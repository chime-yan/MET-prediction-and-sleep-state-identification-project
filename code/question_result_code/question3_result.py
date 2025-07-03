import glob
import os
import re
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from code.data_processing_code.data_normalization_and_encoding import normalization
from code.data_processing_code.data_outlier_detection import outlier_detection
from code.train_val_and_test_model_code.lstm_model import predict, AttentionLSTMWithMLP


def sleep_state_statistics(data_df):
    data_df['time'] = pd.to_datetime(data_df['time'])

    # 计算 NREM 睡眠状态时间
    time_df = data_df.loc[data_df['sleep_state'] == 'NREM', ['time']].sort_values(by=['time'])

    time_df['time_diff'] = time_df['time'].diff()

    # 过滤掉异常时间差（例如，超过1秒的时间差）
    max_allowed_diff = pd.Timedelta(seconds=1)
    valid_diffs = time_df['time_diff'][time_df['time_diff'] <= max_allowed_diff]

    total_time_diff = valid_diffs.sum()
    nrem_time = round(total_time_diff.total_seconds() / 60 / 60, 4)

    # 计算 REM 睡眠状态时间
    time_df = data_df.loc[data_df['sleep_state'] == 'REM', ['time']].sort_values(by=['time'])

    time_df['time_diff'] = time_df['time'].diff()

    # 过滤掉异常时间差（例如，超过1秒的时间差）
    max_allowed_diff = pd.Timedelta(seconds=1)
    valid_diffs = time_df['time_diff'][time_df['time_diff'] <= max_allowed_diff]

    total_time_diff = valid_diffs.sum()
    rem_time = round(total_time_diff.total_seconds() / 60 / 60, 4)

    # 记录睡眠总时长（小时）
    sleep_overall_time = round(nrem_time + rem_time, 4)

    return sleep_overall_time, nrem_time, rem_time


if __name__ == '__main__':
    total_start_time = time.time()

    # 需要加载的模型路径
    model_path = '../../train_and_val_model_results/ques_3_best_lstm_model.pth'
    class_dict = {0: 'NO_SLEEP', 1: 'NREM', 2: 'REM'}

    # 模型参数
    input_size = 28
    output_size = 3

    hidden_size = 64
    lstm_layers = 1
    seq_len = 180

    lstm_dropout = 0
    attn_dropout = 0.1

    mlp_hidden_layers = 2
    mlp_hidden_neurons = [64, 32]
    mlp_activation_fn = nn.GELU()

    mlp_dropout_way = True
    mlp_dropout = 0.2


    # 设置线程数
    # torch.set_num_threads(10)  # 设置用于计算的线程数

    # 定义一个新的模型实例
    loaded_model = AttentionLSTMWithMLP(input_size, output_size, hidden_size, mlp_hidden_layers, mlp_hidden_neurons,
                                 mlp_activation_fn, mlp_dropout_way=mlp_dropout_way, mlp_dropout=mlp_dropout,
                                 lstm_dropout=lstm_dropout, diagonal_projection=False, lstm_layers=lstm_layers,
                                 attn_dropout=attn_dropout, attn_mask=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # 加载模型
    checkpoint = torch.load(model_path, weights_only=True, map_location=torch.device(device))

    # 创建一个新的状态字典，去除 "module." 前缀（使用cpu预测时用，或一个gpu时用）
    new_state_dict = {}
    for key, value in checkpoint['model'].items():
        new_key = key.replace("module.", "")  # 去除 "module." 前缀
        new_state_dict[new_key] = value

    loaded_model.load_state_dict(new_state_dict)

    # 检查是否有多个 GPU
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 个 GPU!")
        loaded_model = nn.DataParallel(loaded_model)  # 封装为 DataParallel

    loaded_model = loaded_model.to(device)


    # 预测睡眠状态

    # 获取文件夹下所有 CSV 文件的路径
    file_list = glob.glob('../../附件2/*.csv')
    # file_list = glob.glob('../../B题-测试数据/*.csv')

    # 定义正则表达式
    pattern = re.compile(r'^(P\d+)\.csv$')

    # 过滤出符合条件的csv文件
    file_list = [f for f in file_list if pattern.match(os.path.basename(f))]

    des_est_df = pd.read_csv(
        '../../dataset_distribution/original_data_distribution/train_data_descriptive_statistics.csv', index_col=0)

    accL2_MET_lower = des_est_df.loc['accL2_MET', 'outlier_lower_quantile']
    accL2_MET_upper = des_est_df.loc['accL2_MET', 'outlier_upper_quantile']

    log_accL2_MET_median = des_est_df.loc['log_accL2_MET', 'median']
    log_accL2_MET_iqr = des_est_df.loc['log_accL2_MET', 'iqr']

    met_lower = des_est_df.loc['MET', 'outlier_lower_quantile']
    met_upper = des_est_df.loc['MET', 'outlier_upper_quantile']

    met_median = des_est_df.loc['MET', 'median']
    met_iqr = des_est_df.loc['MET', 'iqr']


    usecols = ['hour_sin', 'hour_cos', 'minute_sin', 'minute_cos',
               'norm_x', 'norm_y', 'norm_z',
               'sex_F', 'sex_M', 'age_18-29', 'age_30-37', 'age_38-52', 'age_53+',
               'norm_boxcox_acc_L2', 'norm_acc_xy_L2', 'norm_acc_z_abs',
               'is_outlier_x', 'is_outlier_y', 'is_outlier_z',
               'is_outlier_acc_L2', 'is_outlier_acc_xy_L2', 'is_outlier_acc_z_abs',
               'is_midnight', 'is_weekend', 'is_outlier_MET', 'norm_MET',
               'is_outlier_accL2_MET', 'norm_accL2_MET']

    df_all = pd.DataFrame()
    # 流式读取志愿者数据
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

        data_df = pd.read_csv(file_path)

        data_df['is_outlier_MET'] = 0
        met_abnormal_index = outlier_detection(data_df, 'MET',
                                               met_lower, met_upper)
        data_df.loc[met_abnormal_index, 'is_outlier_MET'] = 1

        data_df['norm_MET'] = normalization(data_df['MET'], met_median, met_iqr, alpha=1.0 / 3.0)

        data_df['accL2_MET'] = data_df['acc_L2'] * data_df['MET']
        data_df['is_outlier_accL2_MET'] = 0

        accL2_MET_abnormal_index = outlier_detection(data_df, 'accL2_MET',
                                                     accL2_MET_lower, accL2_MET_upper)
        data_df.loc[accL2_MET_abnormal_index, 'is_outlier_accL2_MET'] = 1

        data_df['log_accL2_MET'] = data_df['accL2_MET'].transform(lambda x: np.log(x + 0.1))
        data_df['norm_accL2_MET'] = normalization(data_df['log_accL2_MET'], log_accL2_MET_median,
                                                  log_accL2_MET_iqr, alpha=1.0 / 1.6)

        data_df['time'] = pd.to_datetime(data_df['time'], format='mixed')
        data_df.sort_values(by=['time'], inplace=True)

        data_df.to_csv(file_path, index=False)


        chunksize = 180
        pred_list = []
        pred_hidden = None
        for chunk in pd.read_csv(file_path, chunksize=chunksize, usecols=usecols):
            chunk = chunk[usecols]  # 调整列顺序

            # 如果当前批次大小小于 chunksize，截取隐藏状态的最后一部分
            if len(chunk) < chunksize:
                pred_hidden = (
                    pred_hidden[0][:, :len(chunk), :],
                    pred_hidden[1][:, :len(chunk), :]
                )

            pred_x = chunk.values

            pred, pred_hidden = predict(loaded_model, device, pred_x,
                                                   question_type='classification', hidden_func=True,
                                                   pred_hidden=pred_hidden, return_sequences=True)

            pred_list.append(pred)

        pred_vector = np.concatenate(pred_list, axis=0)  # 变成 1D 数组

        # 转换为类型名称
        predicted_classes = [class_dict[idx] for idx in pred_vector]

        data_df['sleep_state'] = predicted_classes
        data_df.to_csv(file_path, index=False)

        sleep_overall_time, nrem_time, rem_time = sleep_state_statistics(data_df)
        df = pd.DataFrame({'志愿者ID': [extracted_part],
                           '睡眠总时长（小时）': [f'{sleep_overall_time:.4f}'],
                           'NREM 总时长（小时）': [f'{nrem_time:.4f}'],
                           'REM 总时长（小时）': [f'{rem_time:.4f}']})

        # 追加到总 DataFrame
        df_all = pd.concat([df_all, df], ignore_index=True)

        end_time = time.time()
        one_file_time = end_time - start_time
        print(f'统计{extracted_part}文件的时长数据完成，用了{one_file_time / 60:.2f}分钟')

    df_all.to_excel('../../output_and_result_files/result_3.xlsx', index=False)
    # df_all.to_excel('../../output_and_result_files/result3（测试）.xlsx', index=False)

    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    print(f'统计所有文件的时长数据完成！共使用{total_time / 60:.2f}分钟！')
    print('睡眠状态预测和整理完成！')
