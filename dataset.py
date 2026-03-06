import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import os
import datetime

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

def date_to_ordinal(date):
    # 使用datetime.date.toordinal()方法获取从公元元年1月1日起的天数
    return date.toordinal() - datetime.date(1, 1, 1).toordinal() + 1  # 加1是为了从1开始计数而不是0

# 数据预处理函数 - 针对已统计好的日期-借书量数据
def load_and_process_file(file_path):
    """加载单个CSV文件（已按日期统计借书量）并处理"""
    try:
        # 处理路径中的反斜杠问题
        file_path = file_path.replace('\\', '/')

        # 读取CSV文件，假设列名为'日期'和'借书量'
        df = pd.read_csv(file_path)

        # 转换日期格式
        df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
        df = df.dropna(subset=['日期'])  # 移除日期格式错误的记录

        # 按日期排序
        df = df.sort_values('日期').reset_index(drop=True)

        # 转换为日期类型（不带时间）
        df['日期'] = df['日期'].dt.date

        # # 确保没有重复日期，如果有则取平均值
        # if df['日期'].duplicated().any():
        #     print(f"警告：文件 {file_path} 存在重复日期，已合并")
        #     df = df.groupby('日期')['借书量'].mean().reset_index()
        #     df['借书量'] = df['借书量'].round().astype(int)  # 借书量应为整数
        #
        # # 填补缺失日期
        # start_date = df['日期'].min()
        # end_date = df['日期'].max()
        # all_dates = pd.date_range(start=start_date, end=end_date).date
        # date_df = pd.DataFrame({'日期': all_dates})
        #
        # # 合并并填充缺失值为0
        # daily_counts = pd.merge(date_df, df, on='日期', how='left')
        # daily_counts['借书量'] = daily_counts['借书量'].fillna(0).astype(int)

        return df

    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
        return None


# 自定义数据集
class LibraryDataset(Dataset):
    """图书馆借书量时间序列数据集"""

    def __init__(self, data, input_window=14, output_window=14, step=7, scaler=None, normalize=True):
        """
        参数:
            data: 包含'日期'和'借书量'的DataFrame
            input_window: 输入窗口大小（前14天）
            output_window: 输出窗口大小（预测后14天）
            step: 步长间隔
            scaler: 外部传入的归一化器，用于测试集（与训练集保持一致）
            normalize: 是否归一化数据
        """
        self.data = data
        self.input_window = input_window
        self.output_window = output_window
        self.step = step

        # 提取借书量数据
        self.borrow_counts = self.data['借书量'].values.reshape(-1, 1)
        self.borrow_day = self.data['日期'].values.reshape(-1, 1)

        # 数据归一化
        self.normalize = normalize
        self.scaler = scaler

        if normalize:
            if self.scaler is None:
                # 为训练集创建新的scaler
                self.scaler = MinMaxScaler(feature_range=(0, 1))
                self.borrow_counts = self.scaler.fit_transform(self.borrow_counts)
            else:
                # 为验证集/测试集使用已有的scaler
                self.borrow_counts = self.scaler.transform(self.borrow_counts)

        # 生成样本索引
        self.indices = self._generate_indices()
        print(f"生成了 {len(self.indices)} 个有效样本 (输入窗口: {input_window}, 输出窗口: {output_window})")

    def _generate_indices(self):
        """生成样本的起始索引，确保每个样本都有完整的输入和输出窗口"""
        indices = []
        # 计算最大起始索引，确保有足够的数据用于输入和输出窗口
        max_start_idx = len(self.borrow_counts) - self.input_window - self.output_window-1
        print('len(self.borrow_counts):',len(self.borrow_counts))
        print('max_start_idx:', max_start_idx)

        # 如果数据量不足，返回空列表
        if max_start_idx < 0:
            print(f"警告：数据长度不足，需要至少 {self.input_window + self.output_window} 个数据点，"
                  f"但只找到 {len(self.borrow_counts)} 个")
            return []

        # start_idx = 0
        # while start_idx <= max_start_idx:
        #     indices.append(start_idx)
        #     start_idx += self.step
        # print('indices:', indices)
        start_idx = 0
        while start_idx <= max_start_idx:
            indices.append(start_idx)
            start_idx += 1
        # print('indices:', indices)
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start_idx = self.indices[idx]

        # 提取输入序列（前input_window天）
        input_end = start_idx + self.input_window
        input_seq = self.borrow_counts[start_idx:input_end]

        # 提取目标序列（后output_window天）
        target_end = input_end + self.output_window
        target_seq = self.borrow_counts[input_end:target_end]


        # 日期
        time = self.borrow_day[input_end:target_end][0][0]
        ordinal_date = date_to_ordinal(time)
        time_tensor = torch.tensor([ordinal_date], dtype=torch.long)  # 使用long类型存储天数

        # 转换为PyTorch张量
        input_tensor = torch.FloatTensor(input_seq)
        target_tensor = torch.FloatTensor(target_seq)
        # print('ordinal_date time:',ordinal_date)
        # print('target_tensor =:',target_tensor)

        return time_tensor, input_tensor, target_tensor

    def inverse_transform(self, data):
        """将归一化的数据转换回原始尺度"""
        if self.normalize and self.scaler is not None:
            # 确保数据是二维的(n_samples, n_features)
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
            return self.scaler.inverse_transform(data).squeeze()
        return data

def load_data(train_paths, test_paths, input_window = 14, output_window=14,step=2):
    # 加载并处理训练验证数据（前3个文件）
    print("加载训练验证数据...")
    train_val_dfs = []
    for path in train_paths:
        df = load_and_process_file(path)
        train_val_dfs.append(df)

    # 合并训练验证数据
    train_val_data = pd.concat(train_val_dfs, ignore_index=True)
    # 去重并按日期排序
    # train_val_data = train_val_data.drop_duplicates('日期').sort_values('日期').reset_index(drop=True)
    print(f"训练验证数据时间范围: {train_val_data['日期'].min()} 至 {train_val_data['日期'].max()}")
    print(f"训练验证数据记录数: {len(train_val_data)}")

    # 加载测试数据
    print("\n加载测试数据...")

    test_data = load_and_process_file(test_paths[0])
    print(f"测试数据时间范围: {test_data['日期'].min()} 至 {test_data['日期'].max()}")
    print(f"测试数据记录数: {len(test_data)}")


    # 创建训练验证数据集（使用其scaler来标准化所有数据）
    train_val_dataset = LibraryDataset(
        data=train_val_data,
        input_window=input_window,
        output_window=output_window,
        step=step,
        scaler=None,  # 使用训练集的scaler

        normalize=True
    )
    test_dataset = LibraryDataset(
        data=test_data,
        input_window=input_window,
        output_window=output_window,
        step=step,
        scaler=None,  # 使用训练集的scaler
        normalize=True
    )

    # 划分训练集和验证集 (8:2)
    train_size = int(0.8 * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size
    train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])


    return train_dataset, val_dataset, test_dataset


def load_testdata(test_paths, input_window = 14, output_window=14,step=2):

    # 加载测试数据
    print("\n加载测试数据...")

    test_data = load_and_process_file(test_paths[0])
    print(f"测试数据时间范围: {test_data['日期'].min()} 至 {test_data['日期'].max()}")
    print(f"测试数据记录数: {len(test_data)}")


    test_dataset = LibraryDataset(
        data=test_data,
        input_window=input_window,
        output_window=output_window,
        step=step,
        scaler=None,  # 使用训练集的scaler
        normalize=True
    )

    return test_dataset



if __name__ == "__main__":
    # 文件路径设置 - 前3个为训练验证数据，第4个为测试数据
    # 请根据实际情况修改这些路径
    file_paths = [
        'E:\my_other\other/2025\library\dataset\DataDayCount/Daycount2014.csv',
        'E:\my_other\other/2025\library\dataset\DataDayCount/Daycount2015.csv',
        'E:\my_other\other/2025\library\dataset\DataDayCount/Daycount2016.csv',
        'E:\my_other\other/2025\library\dataset\DataDayCount/Daycount2017.csv',
    ]

    # 检查文件数量
    if len(file_paths) != 4:
        print("错误：需要提供4个文件路径（前3个为训练验证集，第4个为测试集）")
        # return

    # 加载并处理训练验证数据（前3个文件）
    print("加载训练验证数据...")
    train_val_dfs = []
    for path in file_paths[:3]:
        df = load_and_process_file(path)
        if df is not None:
            train_val_dfs.append(df)

    if not train_val_dfs:
        print("错误：无法加载有效的训练验证数据")
        # return

    # 合并训练验证数据
    train_val_data = pd.concat(train_val_dfs, ignore_index=True)
    # 去重并按日期排序
    train_val_data = train_val_data.drop_duplicates('日期').sort_values('日期').reset_index(drop=True)
    print(f"训练验证数据时间范围: {train_val_data['日期'].min()} 至 {train_val_data['日期'].max()}")
    print(f"训练验证数据记录数: {len(train_val_data)}")

    # 加载测试数据（第4个文件）
    print("\n加载测试数据...")
    test_data = load_and_process_file(file_paths[3])
    if test_data is None:
        print("错误：无法加载有效的测试数据")
        # return

    print(f"测试数据时间范围: {test_data['日期'].min()} 至 {test_data['日期'].max()}")
    print(f"测试数据记录数: {len(test_data)}")

    input_window = 14
    output_window = 1
    step = 13
    # 创建训练验证数据集（使用其scaler来标准化所有数据）
    try:
        train_val_dataset = LibraryDataset(
            data=train_val_data,
            input_window=input_window,
            output_window=output_window,
            step=step,
            normalize=True
        )
    except ValueError as e:
        print(f"创建训练验证数据集失败: {e}")
        # return

    # 划分训练集和验证集 (8:2)
    train_size = int(0.8 * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size
    train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])

    # 使用训练集的scaler创建测试数据集
    try:
        test_dataset = LibraryDataset(
            data=test_data,
            input_window=input_window,
            output_window=output_window,
            step=step,
            scaler=train_val_dataset.scaler,  # 使用训练集的scaler
            normalize=True
        )
    except ValueError as e:
        print(f"创建测试数据集失败: {e}")
        # return

    # 创建数据加载器
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"\n训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")


