import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from dataset import load_and_process_file, LibraryDataset, load_data
from model import TransformerTimeSeries
import matplotlib

plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


# 绘制预测结果
def plot_results(model, dataset, device, title, num_samples=5):

    model.eval()
    # 确定实际绘制的样本数（不超过数据集长度）
    actual_samples = min(num_samples, len(dataset))
    fig, axes = plt.subplots(actual_samples, 1, figsize=(12, 3 * actual_samples))
    if actual_samples == 1:
        axes = [axes]

    # 按顺序选择前actual_samples个样本（不打乱顺序）
    indices = list(range(actual_samples))  # 关键修改：从0开始按顺序选取


    with torch.no_grad():
        for i, idx in enumerate(indices):
            inputs, targets = dataset[idx]
            inputs = inputs.unsqueeze(0).to(device)  # 增加批次维度
            # 模型预测
            outputs = model(inputs)
            print('xxx',inputs.shape,outputs.shape,targets.shape)
            print('xxx',inputs.squeeze().cpu().numpy().shape, outputs.squeeze(0).squeeze(-1).cpu().numpy().shape, targets.squeeze(0).cpu().numpy().shape)

            # 转换回原始尺度
            inputs_original = dataset.inverse_transform(inputs.squeeze().cpu().numpy())
            targets_original = dataset.inverse_transform(targets.squeeze(0).cpu().numpy())
            outputs_original = dataset.inverse_transform(outputs.squeeze(0).squeeze(-1).cpu().numpy())


            # 确保所有数据都是至少1维数组（修复len()错误的关键）
            inputs_original = np.atleast_1d(inputs_original)
            targets_original = np.atleast_1d(targets_original)
            outputs_original = np.atleast_1d(outputs_original)

            # 计算绘图范围
            input_len = len(inputs_original)
            target_len = len(targets_original)
            output_len = len(outputs_original)

    #         # 绘制结果
    #         axes[i].plot(range(input_len), inputs_original,
    #                      label='输入（历史14天）', color='blue', linestyle='--')
    #         axes[i].plot(range(input_len, input_len + target_len),
    #                      targets_original, label='真实值', color='green')
    #         axes[i].plot(range(input_len, input_len + output_len),
    #                      outputs_original, label='预测值', color='red', linestyle='--')

    #         axes[i].set_title(f'{title} - 样本 {idx + 1}')
    #         axes[i].set_xlabel('天数')
    #         axes[i].set_ylabel('借书量')
    #         axes[i].legend()
    #         axes[i].grid(True, linestyle=':')


    # plt.tight_layout()
    # plt.show()


# 绘制损失曲线
def plot_losses(train_losses, val_losses):
    """绘制训练和验证损失曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='训练损失')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='验证损失')
    plt.title('训练与验证损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('损失值 (MSE)')
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.show()


# 绘制整体测试结果
def plot_test_overview(all_targets, all_outputs):
    """绘制测试集整体预测情况"""
    plt.figure(figsize=(12, 6))
    plt.plot(all_targets[:100], label='真实值', color='green')
    plt.plot(all_outputs[:100], label='预测值', color='red', linestyle='--')
    plt.title('测试集前100个预测点对比')
    plt.xlabel('样本点')
    plt.ylabel('借书量')
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    plt.show()
