import os
import re

import pandas as pd
import torch
from torch import nn

from lstm_model import predict, AttentionLSTMWithMLP
from ques_3_model_train_and_val import WeightedKLDivLoss


if __name__ == '__main__':
    # 需要加载的模型路径
    model_path = '../../train_and_val_model_results/ques_3_best_lstm_model.pth'

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

    # 测试模型泛化能力
    dataset_path_csv_path = '../../output_and_result_files/dataset_paths.csv'

    # 获取测试数据的所有 CSV 文件的路径
    dataset_path = pd.read_csv(dataset_path_csv_path)
    test_file_list = list(dataset_path.loc[dataset_path['type'] == 'test', 'file_path'])

    loss_fn = WeightedKLDivLoss()

    # 定义正则表达式
    pattern = re.compile(r'^(P\d+)\.csv$')

    test_files_total_loss = 0
    test_files_count = len(test_file_list)

    usecols = ['hour_sin', 'hour_cos', 'minute_sin', 'minute_cos',
               'norm_x', 'norm_y', 'norm_z',
               'sex_F', 'sex_M', 'age_18-29', 'age_30-37', 'age_38-52', 'age_53+',
               'norm_boxcox_acc_L2', 'norm_acc_xy_L2', 'norm_acc_z_abs',
               'is_outlier_x', 'is_outlier_y', 'is_outlier_z',
               'is_outlier_acc_L2', 'is_outlier_acc_xy_L2', 'is_outlier_acc_z_abs',
               'is_midnight', 'is_weekend', 'is_outlier_MET', 'norm_MET',
               'is_outlier_accL2_MET', 'norm_accL2_MET',
               'state_NO_SLEEP', 'state_NREM', 'state_REM']

    # 流式读取志愿者数据
    for file_path in test_file_list:
        # 从文件路径中提取完整文件名
        filename = os.path.basename(file_path)

        # 使用正则查找匹配项
        match = pattern.search(filename)
        if match:
            extracted_part = match.group(1)  # 获取匹配的部分
        else:
            extracted_part = 'Unknown'

        # os.makedirs('./classification_test_results', exist_ok=True)
        # # 动态生成不同文件名
        # test_csv_filename = f"./classification_test_results/{extracted_part}_test_results.csv"


        chunksize = 180
        pred_loss_list = []
        examples = 0

        pred_hidden = None
        for chunk in pd.read_csv(file_path, chunksize=chunksize, usecols=usecols):
            chunk = chunk[usecols]  # 调整列顺序

            # 如果当前批次大小小于 chunksize，截取隐藏状态的最后一部分
            if len(chunk) < chunksize:
                pred_hidden = (
                    pred_hidden[0][:, :len(chunk), :],
                    pred_hidden[1][:, :len(chunk), :]
                )

            test_x = chunk.iloc[:, :-3].values
            test_y = chunk.iloc[:, -3:].values


            _, _, pred_loss, pred_hidden = predict(loaded_model, device, test_x, test_y, loss_fn,
                                            question_type='classification', hidden_func=True,
                                            pred_hidden=pred_hidden, is_soft_labels=True,
                                            return_sequences=True)

            batch_count = test_x.shape[0]
            pred_loss_list.append(pred_loss * batch_count)
            examples += batch_count

            # # 组织数据
            # batch_df = pd.DataFrame({
            #     "prob_0": pred_prob_out[:, 0],
            #     "prob_1": pred_prob_out[:, 1],
            #     "prob_2": pred_prob_out[:, 2],
            #     "targets": test_y,
            #     "pred_targets": pred_y
            # })
            #
            # # 流式写入csv
            # if chunk_count == 0:
            #     batch_df.to_csv(test_csv_filename, index=False, mode="w")  # 第一个 chunk 写入，带表头
            #
            # else:
            #     batch_df.to_csv(test_csv_filename, index=False, mode="a", header=False)  # 追加写入，不重复写表头
            # print(f"Chunk {chunk_count + 1} 的 pred_prob_out 和 test_y 写入 {test_csv_filename}")

        pred_loss = sum(pred_loss_list) / examples
        test_files_total_loss += pred_loss

    test_loss = test_files_total_loss / test_files_count
    print('-------------------------------------------------')
    print(f'Test Loss: {test_loss}')
