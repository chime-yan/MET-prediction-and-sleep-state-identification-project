import pandas as pd
import torch
from torch import nn

from code.data_processing_code.data_normalization_and_encoding import met_anti_norm
from lstm_model import predict, AttentionLSTMWithMLP


if __name__ == '__main__':
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

    loss_fn = nn.L1Loss()

    des_est_df = pd.read_csv(
        '../../dataset_distribution/original_data_distribution/train_data_descriptive_statistics.csv', index_col=0)
    met_median = des_est_df.loc['MET', 'median']
    met_iqr = des_est_df.loc['MET', 'iqr']


    test_files_total_loss = 0
    test_files_count = len(test_file_list)

    usecols = ['hour_sin', 'hour_cos', 'minute_sin', 'minute_cos', 'norm_x', 'norm_y', 'norm_z',
                            'sex_F', 'sex_M', 'age_18-29', 'age_30-37', 'age_38-52', 'age_53+',
                            'norm_boxcox_acc_L2', 'norm_acc_xy_L2', 'norm_acc_z_abs',
                            'is_outlier_x', 'is_outlier_y', 'is_outlier_z',
                            'is_outlier_acc_L2', 'is_outlier_acc_xy_L2', 'is_outlier_acc_z_abs',
                            'is_midnight', 'is_weekend', 'MET']

    for test_file_path in test_file_list:
        chunksize = 180
        pred_loss_list = []
        examples = 0

        pred_hidden = None
        # 读取志愿者数据
        for chunk in pd.read_csv(test_file_path, chunksize=chunksize,
                              usecols=usecols):
            chunk = chunk[usecols]  # 调整列顺序

            # 如果当前批次大小小于 chunksize，截取隐藏状态的最后一部分
            if len(chunk) < chunksize:
                pred_hidden = (
                    pred_hidden[0][:, :len(chunk), :],
                    pred_hidden[1][:, :len(chunk), :]
                )

            test_x = chunk.iloc[:, :-1].values

            # 映射回原始空间进行评估
            test_y = chunk['MET'].values

            _, pred_loss, pred_hidden = predict(loaded_model, device, test_x, test_y, loss_fn,
                                                hidden_func=True, pred_hidden=pred_hidden,
                                                return_sequences=True,
                                                anti_transform_func=met_anti_norm,
                                                column_median=met_median, column_iqr=met_iqr)

            batch_count = test_x.shape[0]
            pred_loss_list.append(pred_loss * batch_count)
            examples += batch_count

        pred_loss = sum(pred_loss_list) / examples
        test_files_total_loss += pred_loss

    test_loss = test_files_total_loss / test_files_count
    print('-------------------------------------------------')
    print(f'Test Loss: {test_loss}')
