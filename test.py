# 导入必要的库
import torch
from model import RNNModel
from poem_data_processing import process_poems
import numpy as np

# 定义开始和结束标记
start_token = 'B'
end_token = 'E'
# 模型保存的目录
model_dir = './model/'
# 诗歌数据文件路径
poems_file = './data/poems.txt'

# 学习率
lr = 0.0002

def to_word(predict, vocabs):
    """
    将预测结果转换为词汇表中的字。

    参数:
    predict: 模型的预测结果，一个概率分布。
    vocabs: 词汇表，包含所有可能的字。

    返回:
    从预测结果中随机选择的一个字。
    """
    predict = predict.numpy()[0]
    predict /= np.sum(predict)
    sample = np.random.choice(np.arange(len(predict)), p=predict)
    if sample > len(vocabs):
        return vocabs[-1]
    else:
        return vocabs[sample]

def gen_poem(begin_word):
    """
    生成诗歌。

    参数:
    begin_word: 诗歌的第一个字。

    返回:
    生成的诗歌，以字符串形式返回。
    """
    batch_size = 1
    # 处理诗歌数据，得到诗歌向量、字到索引的映射和索引到字的映射
    poems_vector, word_to_idx, idx_to_word = process_poems(poems_file)

    # 初始化模型
    model = RNNModel(len(idx_to_word), 128, num_layers=2)
    # 加载模型参数
    checkpoint = torch.load(f'{model_dir}/torch-latest.pth')
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    # 初始化输入序列
    x = torch.tensor([word_to_idx[start_token]], dtype=torch.long).view(1, 1)
    hidden = None

    # 生成诗歌
    with torch.no_grad():
        output, hidden = model(x, hidden)
        predict = torch.softmax(output, dim=1)
        word = begin_word or to_word(predict, idx_to_word)
        poem_ = ''

        i = 0
        while word != end_token:
            poem_ += word
            i += 1
            if i > 24:
                break
            try:
                x = torch.tensor([word_to_idx[word]], dtype=torch.long).view(1, 1)
                output, hidden = model(x, hidden)
                predict = torch.softmax(output, dim=1)
                word = to_word(predict, idx_to_word)
            except KeyError:
              print("很抱歉，我们数据集中不存在你输入的字符，请换一个字！！！")
              exit()


        return poem_

def pretty_print_poem(poem_):
    """
    格式化打印诗歌。

    参数:
    poem_: 生成的诗歌，以字符串形式输入。
    """
    poem_sentences = poem_.split('。')
    for s in poem_sentences:
        if s != '' and len(s) > 10:
            print(s + '。')

if __name__ == '__main__':
    # 用户输入第一个字
    begin_char = input('请输入第一个字 please input the first character: \n')
    print('AI作诗 generating poem...')
    # 生成诗歌
    poem = gen_poem(begin_char)
    # 打印诗歌
    pretty_print_poem(poem_=poem)
