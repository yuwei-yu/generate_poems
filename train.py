import torch
from model import RNNModel
from torch import nn
from poem_data_processing import *
import os
import time

# 检查是否有可用的GPU，如果没有则使用CPU
# windows用户使用torch.cuda.is_available()来检查是否有可用的GPU。
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print(f"Using device: {device}")

def train(poems_path, num_epochs, batch_size, lr):
    """
    训练RNN模型并进行预测。

    参数:
    poems_path (str): 诗歌数据文件路径。
    num_epochs (int): 训练的轮数。
    batch_size (int): 批次大小。
    lr (float): 学习率。
    """
    # 确保模型保存目录存在
    if not os.path.exists('./model'):
        os.makedirs('./model')

    # 处理诗歌数据，生成向量化表示和映射字典
    poems_vector, word_to_idx, idx_to_word = process_poems(poems_path)
    # 初始化RNN模型并将其移动到指定设备
    model = RNNModel(len(idx_to_word), 128, num_layers=2).to(device)
    # 使用Adam优化器初始化训练器
    trainer = torch.optim.Adam(model.parameters(), lr=lr)
    # 使用交叉熵损失函数
    loss_fn = nn.CrossEntropyLoss()

    # 开始训练过程
    for epoch in range(num_epochs):
        loss_sum = 0
        start = time.time()

        # 生成并迭代训练批次
        for X, Y in generate_batch(batch_size, poems_vector, word_to_idx):
            # 将输入和目标数据移动到指定设备
            X = X.to(device)
            Y = Y.to(device)

            state = None
            # 前向传播
            outputs, state = model(X, state)
            Y = Y.view(-1)
            # 计算损失
            l = loss_fn(outputs, Y.long())
            # 反向传播和优化
            trainer.zero_grad()
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
            trainer.step()
            loss_sum += l.item() * Y.shape[0]

        end = time.time()
        print(f"Time cost: {end - start}s")
        print(f"epoch: {epoch}, loss: {loss_sum / (len(poems_vector)*len(poems_vector[0]) )}")

    # 保存模型和优化器的状态
    try:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': trainer.state_dict(),
        }, os.path.join('./model', 'torch-latest.pth'))
    except Exception as e:
        print(f"Error saving model: {e}")

if __name__ == "__main__":
    file_path = "./data/poems.txt"
    train(file_path, num_epochs=100, batch_size=64, lr=0.002)
