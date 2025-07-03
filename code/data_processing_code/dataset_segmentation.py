import os
import random

import pandas as pd
from sklearn.model_selection import train_test_split


def sample_train_files(metadata_df, train_file_paths: list, n_samples: int = 10):
    """
    从训练集中进行基于性别+年龄的 stratified sampling，返回抽样后的文件路径。
    """
    # 筛选出训练集对应的 metadata 行
    train_meta = metadata_df[metadata_df['file_path'].isin(train_file_paths)]

    meta_1_count = train_meta.loc[metadata_df['分层标签'] == 'M_38-52', :]
    train_meta = (metadata_df.drop(metadata_df.loc[metadata_df['分层标签'] == 'M_38-52'].index)
                  .reset_index(drop=True))

    # stratify 标签
    labels = train_meta['sex'].astype(str) + '_' + train_meta['age'].astype(str)

    # stratified sampling（抽样一部分 CSV）
    sampled_meta, _ = train_test_split(
        train_meta,
        train_size=n_samples - 1,
        stratify=labels,
        random_state=None  # 每轮不同抽样
    )

    sampled_meta = pd.concat([sampled_meta, meta_1_count], ignore_index=True)

    return list(sampled_meta['file_path'])


if __name__ == '__main__':
    # 读取 Metadata 文件
    metadata_path = '../../附件1/Metadata1.csv'
    metadata = pd.read_csv(metadata_path)


    # 读取文件路径
    metadata['file_path'] = metadata['pid'].apply(lambda x: os.path.join('../../附件1', x + '.csv'))

    metadata['分层标签'] = metadata['sex'].astype(str) + '_' + metadata['age'].astype(str)
    print(metadata['分层标签'].value_counts())
    print('---------------------------------------')
    print(metadata['age'].value_counts())
    print('---------------------------------------')


    # 将只有3个的 M_38-52 数据取出
    # 将只有7个的 M_30-37 数据取出
    path_list_3 = metadata.loc[metadata['分层标签'] == 'M_38-52', 'file_path'].tolist()
    drop_1 = metadata.loc[metadata['分层标签'] == 'M_38-52', :]
    metadata.drop(metadata.loc[metadata['分层标签'] == 'M_38-52'].index, inplace=True)

    path_list_7 = metadata.loc[metadata['分层标签'] == 'M_30-37', 'file_path'].tolist()
    drop_2 = metadata.loc[metadata['分层标签'] == 'M_30-37', :]
    metadata.drop(metadata.loc[metadata['分层标签'] == 'M_30-37'].index, inplace=True)

    # 按照标签进行分层抽样
    # 60% 训练集
    train_files, temp_files = train_test_split(
        metadata['file_path'],
        train_size=0.6,
        stratify=metadata['分层标签'],
        random_state=21
    )

    # 进一步将 temp_files 分成验证集和测试集
    temp_labels = metadata.loc[metadata['file_path'].isin(temp_files), '分层标签']

    # 20% 验证集，20% 测试集
    val_files, test_files = train_test_split(
        temp_files,
        train_size=0.5,
        stratify=temp_labels,
        random_state=21
    )

    train_df = pd.DataFrame({'file_path': train_files, 'type': 'train'})
    val_df = pd.DataFrame({'file_path': val_files, 'type': 'val'})
    test_df = pd.DataFrame({'file_path': test_files, 'type': 'test'})

    # 合并
    dataset = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)

    # 将只有3个的 M_38-52 数据分别放入训练集、验证集、测试集中
    # 将只有7个的 M_30-37 数据分别放入训练集、验证集、测试集中

    # 随机选择 1 个不重复的元素
    random.seed(21)
    random.shuffle(path_list_3)
    random.shuffle(path_list_7)

    select_path = path_list_3.pop()
    dataset.loc[len(dataset)] = [select_path, 'train']
    train_files = train_files.tolist()
    train_files.append(select_path)
    for _ in range(5):
        select_path = path_list_7.pop()
        dataset.loc[len(dataset)] = [select_path, 'train']
        train_files.append(select_path)

    val_files = val_files.tolist()
    select_path = path_list_3.pop()
    dataset.loc[len(dataset)] = [select_path, 'val']
    val_files.append(select_path)
    select_path = path_list_7.pop()
    dataset.loc[len(dataset)] = [select_path, 'val']
    val_files.append(select_path)

    test_files = test_files.tolist()
    select_path = path_list_3.pop()
    dataset.loc[len(dataset)] = [select_path, 'test']
    test_files.append(select_path)
    select_path = path_list_7.pop()
    dataset.loc[len(dataset)] = [select_path, 'test']
    test_files.append(select_path)


    # 保存为 CSV 文件
    dataset.to_csv('../../output_and_result_files/dataset_paths.csv', index=False)

    metadata = pd.concat([metadata, drop_1, drop_2])
    metadata.to_csv('../../附件1/Metadata1.csv', index=False)


    # 输出划分结果，确认划分后的标签分布是否一致
    print(f"训练集文件数: {len(train_files)}")
    print(metadata.loc[metadata['file_path'].isin(train_files), '分层标签'].value_counts().sort_index())

    print('---------------------------------------')
    print(f"验证集文件数: {len(val_files)}")
    print(metadata.loc[metadata['file_path'].isin(val_files), '分层标签'].value_counts().sort_index())

    print('---------------------------------------')
    print(f"测试集文件数: {len(test_files)}")
    print(metadata.loc[metadata['file_path'].isin(test_files), '分层标签'].value_counts().sort_index())
