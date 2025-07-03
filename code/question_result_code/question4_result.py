import glob
import os
import re
import time
import pandas as pd
import numpy as np


if __name__ == '__main__':
    total_start_time = time.time()

    # 获取所有符合规则的 CSV 文件
    file_list = glob.glob('../../附件2/*.csv')
    pattern = re.compile(r'^(P\d+)\.csv$')
    file_list = [f for f in file_list if pattern.match(os.path.basename(f))]

    for i, file_path in enumerate(file_list):
        filename = os.path.basename(file_path)
        match = pattern.search(filename)
        extracted_part = match.group(1) if match else 'Unknown'

        print(f'第{i + 1}个/共{len(file_list)}个，开始处理{extracted_part}')
        start_time = time.time()

        data_df = pd.read_csv(file_path)
        data_df['time'] = pd.to_datetime(data_df['time'])

        # ========== 向量化久坐判断 ==========
        data_df['is_sedentary'] = (data_df['MET'] >= 1.0) & (data_df['MET'] < 1.6)
        data_df['delta_t'] = data_df['time'].diff().dt.total_seconds().fillna(0)
        data_df.loc[~data_df['is_sedentary'], 'delta_t'] = 0    # 非久坐状态排除
        data_df.loc[(data_df['delta_t'] > 1) & (data_df['delta_t'] < 1), 'delta_t'] = 0      # 时间间隔大于1秒和小于1秒的算作异常时间，不计算在内

        # 获取久坐状态
        data_df['sedentary_segment'] = (data_df['delta_t'] != data_df['delta_t'].shift(1)).cumsum()

        # 删除因shift多出的最后一行
        # data_df.drop('delta_t', axis=0, inplace=True)

        # 只保留久坐编号，非久坐置为 NaN
        data_df.loc[data_df['delta_t'] == 0, 'sedentary_segment'] = pd.NA

        data_df['Sedentary_state'] = 'NO'
        SITTING_THRESHOLD = 30 * 60  # 30分钟的阈值

        # 遍历每段久坐
        for seg_id, seg_df in data_df.groupby('sedentary_segment'):
            if pd.isna(seg_id):
                continue

            # 累加久坐时间
            data_df.loc[seg_df.index, 'sedentary_timer'] = data_df.loc[seg_df.index, 'delta_t'].cumsum()

            # 标记久坐状态
            data_df.loc[seg_df.index, 'Sedentary_state'] = np.where(data_df.loc[seg_df.index, 'sedentary_timer'] >= SITTING_THRESHOLD, 'YES', 'NO')

        # 清理中间列
        data_df.drop(columns=['is_sedentary', 'delta_t', 'sedentary_timer', 'sedentary_segment'], inplace=True)

        # 保存回原CSV（覆盖）
        data_df.to_csv(file_path, index=False)

        end_time = time.time()
        print(f'{extracted_part}处理完成，用时{(end_time - start_time) / 60:.2f} 分钟')

    total_end_time = time.time()
    print(f'全部处理完成！总耗时 {(total_end_time - total_start_time) / 60:.2f} 分钟')
