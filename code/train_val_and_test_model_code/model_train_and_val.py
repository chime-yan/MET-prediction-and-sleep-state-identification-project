import os
import time

import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import optim, nn

from code.lr_scheduler_code.FlexibleWarmupScheduler import FlexibleWarmupScheduler
from code.lr_scheduler_code.CosineAnnealingWarmupRestarts import CosineAnnealingWarmupRestarts

from code.data_processing_code.dataset_segmentation import sample_train_files
from lstm_model import AttentionLSTMWithMLP, get_csv_dataloader, train_model, validate_model, EarlyStopping


if __name__ == '__main__':
    total_start_time = time.time()

    # 模型保存路径
    model_save_path = '../../train_and_val_model_results/best_lstm_model.pth'

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

    epochs = 110
    lr = 5e-4
    batch_size = 128
    weight_decay = 1e-4

    patience = 20
    delta = 0.001

    num_workers = 0
    prefetch_factor = None


    # # 设置线程数
    # torch.set_num_threads(10)  # 设置用于计算的线程数

    # 实例化模型
    model = AttentionLSTMWithMLP(input_size, output_size, hidden_size, mlp_hidden_layers, mlp_hidden_neurons,
                                 mlp_activation_fn, mlp_dropout_way=mlp_dropout_way, mlp_dropout=mlp_dropout,
                                 lstm_dropout=lstm_dropout, diagonal_projection=False, lstm_layers=lstm_layers,
                                 attn_dropout=attn_dropout, attn_mask=True, reg_classifier=True,
                                 classifier_hidden_size=32, classifier_output_size=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # 检查是否有多个 GPU
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 个 GPU!")
        model = nn.DataParallel(model)  # 封装为 DataParallel

    model = model.to(device)

    # 定义优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # optimizer = optim.SGD(
    #     model.parameters(),
    #     lr=lr,
    #     weight_decay=weight_decay,
    #     momentum=0.9,
    #     nesterov=True
    # )

    train_loss_fn = nn.L1Loss()
    vaild_loss_fn = nn.L1Loss()

    # 定义学习率调度器
    warmup_scheduler = FlexibleWarmupScheduler(
        optimizer=optimizer,
        warmup_epochs=int(epochs * 0.05),
        max_lr=lr,
        min_lr=5e-7
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',  # 监测指标是损失（min 表示越小越好）
        factor=0.5,  # 学习率降低的因子
        patience=6,  # 等待 epoch 后没有改善, 就降低学习率
        cooldown=2,  # 学习率降低后，暂停调整 cooldown 个 epoch
        min_lr=5e-7,  # 学习率的下限
        verbose=True  # 是否在学习率调整时打印日志信息
    )

    # total_sample = 5728803
    # total_steps = math.ceil((total_sample / batch_size) * epochs * 0.75)
    # warmup_steps = int(total_steps * 0.1)
    #
    # batch_scheduler = CosineAnnealingWarmupRestarts(
    #     optimizer,
    #     first_cycle_steps=total_steps,
    #     cycle_mult=1,
    #     max_lr=lr,
    #     min_lr=1e-6,
    #     warmup_steps=warmup_steps,
    #     gamma=0.8
    # )

    # 早停法
    early_stopping = EarlyStopping(patience=patience, delta=delta, path=model_save_path)

    # 读取 Metadata 文件
    metadata = pd.read_csv('../../附件1/Metadata1.csv')

    dataset_path_csv_path = '../../output_and_result_files/dataset_paths.csv'

    # 获取训练数据的所有 CSV 文件的路径
    dataset_path = pd.read_csv(dataset_path_csv_path)

    file_list = list(dataset_path.loc[dataset_path['type'] == 'train', 'file_path'])

    # 获取验证数据的所有 CSV 文件的路径
    val_file_list = list(dataset_path.loc[dataset_path['type'] == 'val', 'file_path'])

    train_val_epoch_losses_df = pd.DataFrame(columns=['epoch', 'train_losses', 'val_losses'])
    epoch_losses = []
    epoch_val_losses = []

    # 读取验证集数据
    val_loader = (
        get_csv_dataloader(val_file_list,
                           ['hour_sin', 'hour_cos', 'minute_sin', 'minute_cos', 'norm_x', 'norm_y', 'norm_z',
                            'sex_F', 'sex_M', 'age_18-29', 'age_30-37', 'age_38-52', 'age_53+',
                            'norm_boxcox_acc_L2', 'norm_acc_xy_L2', 'norm_acc_z_abs',
                            'is_outlier_x', 'is_outlier_y', 'is_outlier_z',
                            'is_outlier_acc_L2', 'is_outlier_acc_xy_L2', 'is_outlier_acc_z_abs',
                            'is_midnight', 'is_weekend', 'state_NO_SLEEP', 'norm_MET_target'],
                           batch_size=batch_size, seq_len=seq_len,
                           num_workers=num_workers, prefetch_factor=prefetch_factor, reg_classifier=True))

    for i, epoch in enumerate(range(epochs)):
        print(f'第{i + 1}个 epoch / 共{epochs}个 epoch')
        start_time = time.time()

        # 每一轮训练时，从 file_list 中按“性别+年龄”分层抽样出一部分 CSV 文件，从中加载数据进入训练，
        # 且每一轮 epoch 分层抽样出的csv文件是不同的，以提高模型泛化性能。(抽样40%)
        sampled_files = sample_train_files(metadata, file_list, n_samples=24)

        # 读取训练集数据
        train_loader = (
            get_csv_dataloader(sampled_files,
                               ['hour_sin', 'hour_cos', 'minute_sin', 'minute_cos', 'norm_x', 'norm_y', 'norm_z',
                                'sex_F', 'sex_M', 'age_18-29', 'age_30-37', 'age_38-52', 'age_53+',
                                'norm_boxcox_acc_L2', 'norm_acc_xy_L2', 'norm_acc_z_abs',
                                'is_outlier_x', 'is_outlier_y', 'is_outlier_z',
                                'is_outlier_acc_L2', 'is_outlier_acc_xy_L2', 'is_outlier_acc_z_abs',
                                'is_midnight', 'is_weekend', 'state_NO_SLEEP', 'norm_MET_target'],
                               batch_size=batch_size, seq_len=seq_len,
                               num_workers=num_workers, prefetch_factor=prefetch_factor, reg_classifier=True))

        # 训练模型
        for _ in range(2):
            model, epoch_loss = (
                train_model(model, train_loader, train_loss_fn, optimizer, device, hidden_func=True,
                            reg_classifier=True, classifier_loss_alpha=0.3))

        # 保存一个 epoch 总的平均损失
        epoch_losses.append(epoch_loss)
        print(f'Epoch {epoch + 1}, Loss: {epoch_loss:.4f}')

        # 验证过程
        epoch_val_loss, epoch_val_mu_target_loss = (
            validate_model(model, val_loader, vaild_loss_fn, device, hidden_func=True,
                           reg_mu_target_lower=-0.673077, reg_mu_target_upper=-0.288462,
                           reg_classifier=True, classifier_loss_alpha=0.3))

        if not warmup_scheduler.is_finished():
            warmup_scheduler.step_epoch()
        else:
            scheduler.step(epoch_val_loss)

        epoch_val_losses.append(epoch_val_loss)
        print(f'Epoch {epoch + 1}, Val Loss: {epoch_val_loss:.4f}')
        print(f'Val Mu-Target MAE: {epoch_val_mu_target_loss:.4f}')

        # # 早停检查
        early_stopping(epoch_val_loss, model)

        end_time = time.time()
        one_file_time = end_time - start_time
        print(f'第{i + 1}个 epoch 完成，用了{one_file_time / 60:.2f}分钟')

        # 如果早停触发，停止训练
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    train_val_epoch_losses_df['epoch'] = range(1, epoch + 2)
    train_val_epoch_losses_df['train_losses'] = epoch_losses
    train_val_epoch_losses_df['val_losses'] = epoch_val_losses

    os.makedirs('../../train_and_val_model_results/regression_train_and_val_results', exist_ok=True)
    train_val_epoch_losses_df.to_csv('../../train_and_val_model_results/regression_train_and_val_results/train_val_all_epoch_losses.csv',
                                     index=False)

    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    print(f'训练完成！共使用{total_time / 60:.2f}分钟！')

    # 绘制 Loss 曲线
    epochs = train_val_epoch_losses_df['epoch']
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_val_epoch_losses_df['train_losses'], label="Train Mixed Loss", linestyle='-')
    plt.plot(epochs, train_val_epoch_losses_df['val_losses'], label="Val Mixed Loss", linestyle='-')

    plt.xlim(0, len(epochs) + 1)

    plt.xlabel("Epochs")
    plt.ylabel("Mixed Loss")
    plt.title("Mixed Loss Curve")
    plt.legend()

    plt.savefig('../../train_and_val_model_results/regression_train_and_val_results/Mixed_Loss_Curve.png')
    plt.show()
