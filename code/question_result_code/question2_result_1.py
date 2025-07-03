import glob
import os
import re
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import boxcox
from torch import nn

from code.data_processing_code.data_normalization_and_encoding import normalization, met_anti_norm, feature_extra, \
    encoding
from code.train_val_and_test_model_code.lstm_model import AttentionLSTMWithMLP, predict


if __name__ == '__main__':
    total_start_time = time.time()

    # 需要加载的模型路径
    model_path = '../../train_and_val_model_results/best_lstm_model.pth'

    # 模型参数
    input_size = 24
    output_size = 1

    hidden_size = 64
    lstm_layers = 1
    seq_len = 180

    lstm_dropout = 0
    attn_dropout = 0.1

    mlp_hidden_layers = 2
    mlp_hidden_neurons = [64, 32]
    mlp_activation_fn = nn.GELU()

    mlp_dropout_way = True
    mlp_dropout = 0.1


    # 设置线程数
    # torch.set_num_threads(10)  # 设置用于计算的线程数

    # 定义一个新的模型实例
    loaded_model = AttentionLSTMWithMLP(input_size, output_size, hidden_size, mlp_hidden_layers, mlp_hidden_neurons,
                                 mlp_activation_fn, mlp_dropout_way=mlp_dropout_way, mlp_dropout=mlp_dropout,
                                 lstm_dropout=lstm_dropout, diagonal_projection=False, lstm_layers=lstm_layers,
                                 attn_dropout=attn_dropout, attn_mask=True, reg_classifier=True,
                                 classifier_hidden_size=32, classifier_output_size=2,
                                 used_reg_classifier=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # 加载模型
    checkpoint = torch.load(model_path, weights_only=True, map_location=torch.device(device))

    # 创建一个新的状态字典，去除 "module." 前缀
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


    # 预测MET值

    # 获取文件夹下所有 CSV 文件的路径
    file_list = glob.glob('../../附件2/*.csv')
    # file_list = glob.glob('../../B题-测试数据/*.csv')

    # 定义正则表达式
    pattern = re.compile(r'^(P\d+)\.csv$')

    # 过滤出符合条件的csv文件
    file_list = [f for f in file_list if pattern.match(os.path.basename(f))]


    des_est_df = pd.read_csv(
        '../../dataset_distribution/original_data_distribution/train_data_descriptive_statistics.csv', index_col=0)
    x_acc_median = des_est_df.loc['x', 'median']
    x_acc_iqr = des_est_df.loc['x', 'iqr']

    y_acc_median = des_est_df.loc['y', 'median']
    y_acc_iqr = des_est_df.loc['y', 'iqr']

    z_acc_median = des_est_df.loc['z', 'median']
    z_acc_iqr = des_est_df.loc['z', 'iqr']

    acc_xy_l2_median = des_est_df.loc['acc_xy_L2', 'median']
    acc_xy_l2_iqr = des_est_df.loc['acc_xy_L2', 'iqr']

    acc_z_abs_median = des_est_df.loc['acc_z_abs', 'median']
    acc_z_abs_iqr = des_est_df.loc['acc_z_abs', 'iqr']

    boxcox_acc_l2_median = des_est_df.loc['boxcox_acc_L2', 'median']
    boxcox_acc_l2_iqr = des_est_df.loc['boxcox_acc_L2', 'iqr']

    met_median = des_est_df.loc['MET', 'median']
    met_iqr = des_est_df.loc['MET', 'iqr']


    # 读取 Metadata
    metadata_df = pd.read_csv('../../附件2/Metadata2.csv', low_memory=False)
    # metadata_df = pd.read_csv('../../B题-测试数据/Metadata3.csv', low_memory=False)

    lambda_records = []
    usecols = ['hour_sin', 'hour_cos', 'minute_sin', 'minute_cos', 'norm_x', 'norm_y', 'norm_z',
               'sex_F', 'sex_M', 'age_18-29', 'age_30-37', 'age_38-52', 'age_53+',
               'norm_boxcox_acc_L2', 'norm_acc_xy_L2', 'norm_acc_z_abs',
               'is_outlier_x', 'is_outlier_y', 'is_outlier_z',
               'is_outlier_acc_L2', 'is_outlier_acc_xy_L2', 'is_outlier_acc_z_abs',
               'is_midnight', 'is_weekend']

    for a, file_path in enumerate(file_list):
        # 从文件路径中提取完整文件名
        filename = os.path.basename(file_path)

        # 使用正则查找匹配项
        match = pattern.search(filename)
        if match:
            extracted_part = match.group(1)  # 获取匹配的部分
        else:
            extracted_part = 'Unknown'

        print(f'第{a + 1}个/共{len(file_list)}个，开始处理{extracted_part}')
        start_time = time.time()

        # 读取志愿者数据
        data_df = pd.read_csv(file_path).dropna()

        # 将 Metadata 的年龄和性别数据整理到一起
        data_df['age'] = metadata_df.loc[metadata_df['pid'] == extracted_part, 'age'].iloc[0]
        data_df['sex'] = metadata_df.loc[metadata_df['pid'] == extracted_part, 'sex'].iloc[0]

        data_df = feature_extra(data_df)
        data_df = encoding(data_df)

        data_df['norm_x'] = normalization(data_df['x'], x_acc_median, x_acc_iqr)
        data_df['norm_y'] = normalization(data_df['y'], y_acc_median, y_acc_iqr, alpha=1.0 / 2.5)
        data_df['norm_z'] = normalization(data_df['z'], z_acc_median, z_acc_iqr, alpha=1.0 / 1.2)
        data_df['norm_acc_xy_L2'] = normalization(data_df['acc_xy_L2'], acc_xy_l2_median,
                                                  acc_xy_l2_iqr, alpha=1.0 / 4.2)
        data_df['norm_acc_z_abs'] = normalization(data_df['acc_z_abs'], acc_z_abs_median,
                                                  acc_z_abs_iqr, alpha=1.0 / 1.6)

        boxcox_acc_l2, acc_l2_lambda_opt = boxcox(data_df['acc_L2'].values)
        min_clip = np.quantile(boxcox_acc_l2, 0.05)
        max_clip = np.quantile(boxcox_acc_l2, 0.95)

        data_df['boxcox_acc_L2'] = np.clip(boxcox_acc_l2, a_min=min_clip, a_max=max_clip)

        data_df['norm_boxcox_acc_L2'] = normalization(data_df['boxcox_acc_L2'], boxcox_acc_l2_median,
                                                      boxcox_acc_l2_iqr, alpha=1.0 / 3.0)

        # 构造离散时间段
        data_df['time'] = pd.to_datetime(data_df['time'], format='mixed')
        hour = data_df['time'].dt.hour + data_df['time'].dt.minute / 60

        data_df['is_midnight'] = hour.between(0, 5, inclusive="left").astype(int)
        data_df['is_weekend'] = data_df['time'].dt.weekday.between(5, 6).astype(int)

        data_df.sort_values(by="time", inplace=True, ascending=True)

        data_df['MET'] = 0
        lambda_records.append({'file': filename, 'acc_l2_lambda': acc_l2_lambda_opt})

        chunksize = 180
        num_rows = len(data_df)
        pred_hidden = None
        for i in range(0, num_rows, chunksize):
            end_index = min(i + chunksize - 1, num_rows - 1)
            pred_x = data_df.loc[i:end_index, usecols].values

            # 如果当前批次大小小于 chunksize，截取隐藏状态的最后一部分
            if len(pred_x) < chunksize:
                pred_hidden = (
                    pred_hidden[0][:, :len(pred_x), :],
                    pred_hidden[1][:, :len(pred_x), :]
                )

            pred, pred_hidden = predict(loaded_model, device, pred_x,
                                                hidden_func=True, pred_hidden=pred_hidden,
                                                return_sequences=True)

            pred = met_anti_norm(pred, met_median, met_iqr)
            data_df.loc[i:end_index, 'MET'] = pred

        data_df.to_csv('../../附件2/' + extracted_part + '.csv', index=False)
        # data_df.to_csv('../../B题-测试数据/' + extracted_part + '.csv', index=False)

        data_df.drop(usecols, axis=1, inplace=True)
        data_df.drop(['boxcox_acc_L2', 'acc_xy_L2', 'acc_z_abs', 'acc_L2', 'age', 'sex'], axis=1, inplace=True)

        os.makedirs('../../result_2', exist_ok=True)
        data_df.to_csv('../../result_2/' + extracted_part + '.csv', index=False)

        end_time = time.time()
        one_file_time = end_time - start_time
        print(f'预测{extracted_part}文件的MET完成，用了{one_file_time / 60:.2f}分钟')

    pd.DataFrame(lambda_records).to_csv('../../附件2/acc_l2_lambdas_annex_2.csv', index=False)

    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    print(f'MET预测完成！共使用{total_time / 60:.2f}分钟！')
