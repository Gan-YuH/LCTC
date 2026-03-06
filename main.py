import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from dataset import  load_data
from model import TransformerTimeSeries
from utils import plot_results,plot_losses,plot_test_overview
import os
from test import evaluate_model, save_result

from MyWeight.Min30count_in2048_out1_step1_hs64_nl4_nh4.config import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 训练模型
# 训练模型并保存最佳权重
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50,
                device='cpu', save_path='./MyWeight/'):
    """训练模型，返回训练和验证损失，并保存验证损失最低的模型权重"""
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')  # 初始化最佳验证损失为无穷大

    # 创建保存目录（如果不存在）
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model.to(device)

    for epoch in range(num_epochs):
        # 训练模式
        model.train()
        train_loss = 0.0

        for times, inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # 清零梯度
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        # 计算平均训练损失
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # 验证模式
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for times, inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        # 计算平均验证损失
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        # 打印 epoch 信息
        if (epoch + 1) % 3 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], 训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}')
            save_epoch_path = save_path + 'epoch' + str(epoch) + '.pth'
            torch.save(model.state_dict(), save_epoch_path)

        # 保存验证损失最低的模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_best_path = save_path + 'best.pth'
            torch.save(model.state_dict(), save_best_path)
            print(f'第 {epoch + 1} 轮验证损失最低，已保存模型权重至 {save_path}')

    return train_losses, val_losses

#
# # 在测试集上评估模型
# def evaluate_model(model, test_loader, dataset, device):
#     """在测试集上评估模型并返回评估指标"""
#     model.eval()
#     all_targets = []
#     all_outputs = []
#
#     with torch.no_grad():
#         for times, inputs, targets in test_loader:
#             inputs = inputs.to(device)
#             outputs = model(inputs)
#
#             # 转换回原始尺度
#             targets_original = dataset.inverse_transform(targets.numpy())
#             outputs_original = dataset.inverse_transform(outputs.cpu().numpy())
#
#             all_targets.extend(targets_original)
#             all_outputs.extend(outputs_original)
#
#     # 计算评估指标
#     mae = mean_absolute_error(all_targets, all_outputs)
#     rmse = np.sqrt(mean_squared_error(all_targets, all_outputs))
#
#     # 处理可能的零值，避免除零错误
#     non_zero_mask = np.array(all_targets) != 0
#     if np.any(non_zero_mask):
#         mape = np.mean(np.abs((np.array(all_targets)[non_zero_mask] - np.array(all_outputs)[non_zero_mask]) /
#                               np.array(all_targets)[non_zero_mask])) * 100
#     else:
#         mape = 0.0  # 所有目标值都是零
#
#     print(f"\n测试集评估指标:")
#     print(f"MAE (平均绝对误差): {mae:.2f}")
#     print(f"RMSE (均方根误差): {rmse:.2f}")
#     print(f"MAPE (平均绝对百分比误差): {mape:.2f}%")
#
#     return all_targets, all_outputs



def main():

    train_dataset, val_dataset, test_dataset = load_data(train_paths, test_paths,
                                                      input_window=input_window, output_window=output_window, step=step)

    print(f"\n训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 初始化模型、损失函数和优化器
    model = TransformerTimeSeries(input_size=input_size, input_seq=input_window,
                                  hidden_size=hidden_size, num_layers=num_layers, num_heads=num_heads,
                                  out_seq=output_window)#.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)


    # 检查是否有GPU可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    # 训练模型
    print("\n开始训练...")
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        num_epochs=100, save_path=save_path, device=device
    )

    # 在测试集上评估模型
    model.load_state_dict(torch.load(test_weights_path, map_location=device))
    all_targets, all_outputs = evaluate_model(model, test_loader, test_dataset, device,save_path)
    save_result(test_paths[0], save_path, all_outputs, input_window)





if __name__ == "__main__":
    main()