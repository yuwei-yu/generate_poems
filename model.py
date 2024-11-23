import torch
import torch.nn as nn
import torch.optim as optim


class RNNModel(nn.Module):
    def __init__(self, vocab_size, rnn_size=128, num_layers=2):
        """
        构建RNN序列到序列模型。
        :param vocab_size: 词汇表大小
        :param rnn_size: RNN隐藏层大小
        :param num_layers: RNN层数

        """
        super(RNNModel, self).__init__()

        # 选择LSTM单元
        # 参数说明：输入大小、隐藏层大小、层数、batch_first=True表示输入数据的第一维是批次大小
        self.cell = nn.LSTM(rnn_size, rnn_size, num_layers, batch_first=True)

        # 嵌入层，将词汇表中的词转换为向量
        # vocab_size + 1 是因为在词嵌入中需要有一个特殊标记，用于表示填充位置，所以词嵌入时会加一个词。
        self.embedding = nn.Embedding(vocab_size + 1, rnn_size)

        # RNN隐藏层大小
        self.rnn_size = rnn_size

        # 全连接层，用于输出预测
        # 输入大小为RNN隐藏层大小，输出大小为词汇表大小加1
        self.fc = nn.Linear(rnn_size, vocab_size + 1)

    def forward(self, input_data, hidden):
        """
        前向传播
        :param input_data: 输入数据，形状为 (batch_size, sequence_length)
        :param output_data: 输出数据（训练时提供），形状为 (batch_size, sequence_length)
        :return: 输出结果或损失
        """
        # 获取批次大小
        batch_size = input_data.size(0)

        # 嵌入层，将输入数据转换为向量
        # 输入数据形状为 (batch_size, sequence_length)，嵌入后形状为 (batch_size, sequence_length, rnn_size)
        embedded = self.embedding(input_data)
        # 通过RNN层
        # 输入形状为 (batch_size, sequence_length, rnn_size)，输出形状为 (batch_size, sequence_length, rnn_size)
        outputs, hidden = self.cell(embedded, hidden)
        # 将输出展平
        # 展平后的形状为 (batch_size * sequence_length, rnn_size)
        outputs = outputs.contiguous().view(-1, self.rnn_size)
        # 通过全连接层
        # 输入形状为 (batch_size * sequence_length, rnn_size)，输出形状为 (batch_size * sequence_length, vocab_size + 1)
        logits = self.fc(outputs)
        return logits, hidden
