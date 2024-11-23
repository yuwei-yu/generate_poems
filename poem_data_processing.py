import collections
import numpy as np
import torch

# 定义起始和结束标记
start_token = 'B'
end_token = 'E'

def process_poems(file_name):
    """
    处理诗歌文件，将诗歌转换为数字序列，并构建词汇表。

    :param file_name: 诗歌文件的路径
    :return:
        - poems_vector: 诗歌的数字序列列表
        - word_to_idx: 词汇到索引的映射字典
        - idx_to_word: 索引到词汇的映射列表
    """
    # 初始化诗歌列表
    poems = []

    # 读取文件并处理每一行
    with open(file_name, "r", encoding='utf-8') as f:
        for line in f.readlines():
            try:
                # 分割标题和内容
                title, content = line.strip().split(':')
                content = content.replace(' ', '')

                # 过滤掉包含特殊字符的诗歌
                if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content or \
                        start_token in content or end_token in content:
                    continue

                # 过滤掉长度不符合要求的诗歌
                if len(content) < 5 or len(content) > 79:
                    continue

                # 添加起始和结束标记
                content = start_token + content + end_token
                poems.append(content)
            except ValueError as e:
                pass

    # 统计所有单词的频率
    all_words = [word for poem in poems for word in poem]
    counter = collections.Counter(all_words)
    words = sorted(counter.keys(), key=lambda x: counter[x], reverse=True)

    # 添加空格作为填充符
    words.append(' ')
    words_length = len(words)

    # 构建词汇到索引和索引到词汇的映射
    word_to_idx = {word: i for i, word in enumerate(words)}
    idx_to_word = [word for word in words]

    # 将诗歌转换为数字序列
    poems_vector = [[word_to_idx[word] for word in poem] for poem in poems]

    return poems_vector, word_to_idx, idx_to_word

def generate_batch(batch_size, poems_vec, word_to_int):
    """
    生成批量训练数据。
    :param batch_size: 批量大小
    :param poems_vec: 诗歌的数字序列列表
    :param word_to_int: 词汇到索引的映射字典
    :return:
        - x_batches: 输入数据批次
        - y_batches: 目标数据批次
    """
    # 计算可以生成的批次数
    num_example = len(poems_vec) // batch_size

    x_batches = []
    y_batches = []

    for i in range(num_example):
        start_index = i * batch_size
        end_index = start_index + batch_size

        # 获取当前批次的诗歌
        batches = poems_vec[start_index:end_index]

        # 找到当前批次中最长的诗歌长度
        length = max(map(len, batches))

        # 初始化输入数据，使用空格进行填充
        x_data = np.full((batch_size, length), word_to_int[' '], np.int32)

        # 填充输入数据
        for row, batch in enumerate(batches):
            x_data[row, :len(batch)] = batch

        # 创建目标数据，目标数据是输入数据向右移一位
        y_data = np.copy(x_data)
        y_data[:, :-1] = x_data[:, 1:]

        """
        x_data             y_data
        [6,2,4,6,9]       [2,4,6,9,9]
        [1,4,2,8,5]       [4,2,8,5,5]
        """

        # 将当前批次的数据添加到列表中
        yield torch.tensor(x_data), torch.tensor(y_data)
