import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
import matplotlib.pyplot as plt
from dataset import  load_testdata
from model import TransformerTimeSeries
import matplotlib
# from config import *

from MyWeight.Daycount_in7_out1_step1_hs64_nl4_nh4.config import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # 设置中文显示
# matplotlib.use('TkAgg')


# 在测试集上评估模型
def evaluate_model(model, test_loader, dataset, device, save_path):
    """在测试集上评估模型并返回评估指标"""
    model.eval()
    all_targets = []
    all_outputs = []
    all_time = []
    with torch.no_grad():
        for times, inputs, targets in test_loader:
            inputs = inputs.to(device)
            # outputs = model(inputs,inputs,inputs,inputs)
            outputs = model(inputs,)

            # 转换回原始尺度
            # print('targets:',targets)
            # print('targets2',targets.shape)
            targets_original = dataset.inverse_transform(targets.squeeze(-1).numpy())
            outputs_original = dataset.inverse_transform(outputs.squeeze(-1).cpu().numpy())
            outputs_original = np.where(outputs_original<0, 0, outputs_original)
            outputs_original = np.round(outputs_original)

            all_time.extend(times.squeeze(-1).cpu().numpy())
            all_targets.extend(targets_original.flatten())
            all_outputs.extend(outputs_original.flatten())

    # 计算评估指标
    # 计算归一化所需的最大值和最小值（使用目标值的范围）
    targets_array = np.array(all_targets)
    outputs_array = np.array(all_outputs)

    # 计算目标值的最大最小值用于归一化
    max_val = np.max(targets_array)
    min_val = np.min(targets_array)

    # 处理可能的除零情况（如果所有值都相同）
    if max_val == min_val:
        # 归一化后所有值都为0.5
        normalized_targets = np.full_like(targets_array, 0.5)
        normalized_outputs = np.full_like(outputs_array, 0.5)
    else:
        # 进行Min-Max归一化
        normalized_targets = (targets_array - min_val) / (max_val - min_val)
        normalized_outputs = (outputs_array - min_val) / (max_val - min_val)

    # 使用归一化后的数据计算MAE和RMSE
    mae = mean_absolute_error(normalized_targets, normalized_outputs)
    rmse = np.sqrt(mean_squared_error(normalized_targets, normalized_outputs))

    # 处理可能的零值，避免除零错误（使用原始数据计算MAPE）
    non_zero_mask = targets_array != 0
    if np.any(non_zero_mask):
        mape = np.mean(np.abs((targets_array[non_zero_mask] - outputs_array[non_zero_mask]) /
                            targets_array[non_zero_mask])) * 100
    else:
        mape = 0.0  # 所有目标值都是零

    # ---------- 新增：计算 R² ----------
    if max_val == min_val:
        # 所有目标值相同的情况
        if np.all(normalized_outputs == normalized_targets[0]):
            r2 = 1.0
        else:
            r2 = 0.0  # 无法解释方差，设为 0
    else:
        # 正常情况，使用归一化数据计算 R²（等价于原始数据的 R²）
        r2 = r2_score(normalized_targets, normalized_outputs)

    print(f"\n测试集评估指标 (MAE和RMSE基于归一化数据):")
    print(f"归一化MAE (平均绝对误差): {mae:.4f}")
    print(f"归一化RMSE (均方根误差): {rmse:.4f}")
    print(f"MAPE (平均绝对百分比误差，基于原始数据): {mape:.2f}%")
    print(f"   R-squared (R²): {r2:.4f}")

    # 保存为可读的txt格式
    txt_filename = save_path + 'result.txt'
    with open(txt_filename, 'w', encoding='utf-8') as f:
        f.write("测试集评估指标 (MAE和RMSE基于归一化数据):\n")
        f.write(f"归一化MAE (平均绝对误差): {mae:.4f} \n")
        f.write(f"归一化RMSE (均方根误差): {rmse:.4f}\n")
        f.write(f"MAPE (平均绝对百分比误差，基于原始数据): {mape:.2f}% \n")

        f.write(f"R-squared (R²): {r2:.4f}\n")



    x = range(len(all_targets))
    plt.figure(figsize=(12, 6))
    plt.plot(x, all_targets, label='true', linewidth=2, color='red', markersize=8)
    plt.plot(x, all_outputs, label='pred', linewidth=2, color='green',markersize=8)
    plt.xlabel('index')
    plt.ylabel('value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # plt.xticks(x)
    plt.savefig(save_path + 'pred.jpg')


    return all_targets, all_outputs


# 在测试集上评估模型
def save_result(input_path, out_path, result, seq_len):
    df = pd.read_csv(input_path, usecols=[0, 1])

    list_result = [str(x).strip() for x in result]
    # 处理数据
    third_column = [None] * seq_len + list_result

    if len(third_column) < len(df):
        third_column.extend([None] * (len(df) - len(third_column)))
    elif len(third_column) > len(df):
        third_column = third_column[:len(df)]

    df['pred'] = third_column

    df.to_csv(out_path + 'result.csv', index=False, encoding='utf-8-sig')
    print(f"处理完成！文件已保存到: {out_path}")
    print("\n处理后的数据预览:")
    print(df.head())



def main():



    # 检查是否有GPU可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")


    test_dataset = load_testdata( test_paths, input_window=input_window, output_window=output_window, step=step)
    print(f"测试集样本数: {len(test_dataset)}")

    # 创建数据加载器
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


    # 初始化模型、损失函数和优化器
    model = TransformerTimeSeries(input_size=input_size, input_seq=input_window,
                                  hidden_size=hidden_size, num_layers=num_layers, num_heads=num_heads,
                                  out_seq=output_window).to(device)




    model.load_state_dict(torch.load(test_weights_path, map_location=device))

    all_targets, all_outputs = evaluate_model(model, test_loader, test_dataset, device, save_path)

    save_result(test_paths[0], save_path, all_outputs, input_window)


    # # 训练模型
    # print("\n开始训练...")
    # train_losses, val_losses = train_model(
    #     model, train_loader, val_loader, criterion, optimizer,
    #     num_epochs=10, device=device
    # )
    #
    # # 绘制损失曲线
    # plot_losses(train_losses, val_losses)
    #
    # # 绘制训练集和验证集上的预测结果
    # plot_results(model, train_dataset.dataset, device, "训练集预测结果", num_samples=500)
    # plot_results(model, val_dataset.dataset, device, "验证集预测结果", num_samples=500)
    #
    # # 在测试集上评估模型
    # all_targets, all_outputs = evaluate_model(model, test_loader, test_dataset, device)
    #
    # # 绘制测试集结果
    # plot_results(model, test_dataset, device, "测试集预测结果", num_samples=3)
    # plot_test_overview(all_targets, all_outputs)


if __name__ == "__main__":
    main()