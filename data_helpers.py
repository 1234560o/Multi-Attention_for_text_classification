import numpy as np
import re
from collections import Counter
import tensorflow.contrib.keras as kr
import matplotlib.pyplot as plt
import pandas as pd
from operator import itemgetter


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    # 英文单引号问题''
    string = re.sub(r" \'", " \' ", string)
    string = re.sub(r"\' ", " \' ", string)
    string = re.sub(r"^\'", "\' ", string)
    string = re.sub(r"\'$", " \'", string)

    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)   # \s匹配空格符（包括空格、制表符、换页符等等）
    return string.strip().lower()


# read rt-polarity data and Subj data
def load_polarity_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r", encoding='utf-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r", encoding='utf-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [1 for _ in positive_examples]
    negative_labels = [0 for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels])
    print("读取MR或Subj数据集完成，共有{}条数据".format(len(x_text)))
    print("打印出该数据集第一个文本：{}，第一个标签：{}".format(x_text[0], y[0]))
    print("打印出该数据集第四个文本：{}，第四个标签：{}".format(x_text[3], y[3]))
    print("打印出该数据集最后一个个文本：{}，最后一个个标签：{}"
          .format(x_text[len(x_text)-1], y[len(x_text)-1]))
    return [x_text, y]


def load_data(path):
    x_text = []
    y = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            content = line.strip()
            if content == '':
                continue
            else:
                content = content.split('\t' ,1)
                # content[1] = clean_str(content[1])
                x_text.append(content[1])
                y.append(content[0])
    print("读取数据集完成，共有{}条数据".format(len(x_text)))
    print("打印出该数据集第一个文本：{}，第一个标签：{}".format(x_text[0], y[0]))
    print("打印出该数据集第四个文本：{}，第四个标签：{}".format(x_text[3], y[3]))
    print("打印出该数据集最后一个个文本：{}，最后一个个标签：{}"
          .format(x_text[len(x_text) - 1], y[len(x_text) - 1]))
    return x_text, y


def load_cnews_char_data(path):
    x_text = []
    y = []
    label2index = {'体育': 0, '财经': 1, '房产': 2, '家居': 3,
                   '教育': 4, '科技': 5, '时尚': 6, '时政': 7,
                   '游戏': 8, '娱乐': 9}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            content = line.strip()
            if content == '':
                continue
            else:
                x_text.append(' '.join(list(content.split('\t')[1])))
                y.append(label2index[content.split('\t')[0]])
    return x_text, y


def vision_length(x_text):
    text_length = np.array([len(content.split()) for content in x_text])
    # print(text_length)
    # print(Counter(text_length))
    # print(Counter(text_length).keys())
    text_length=text_length.reshape(-1, 1)
    print(text_length.shape)
    frame = pd.DataFrame(text_length, columns=['length'])
    frame['add'] = pd.DataFrame(np.arange(text_length.shape[0]).reshape(-1, 1))
    # print(frame)
    frame = frame.groupby('length').count()
    print(frame)
    frame.sort_values('length').plot.bar()
    plt.show()


def build_vocab(x_text, vocab_dir, vocab_size):
    """根据训练集构建词汇表，存储"""
    # x_text, _ = load_polarity_data_and_labels(pos_file, neg_file)

    # all_data = []
    # counter = Counter()
    counter = {}
    for content in x_text:
        # all_data.extend(content.split())
        for word in content.split():
            counter[word] = counter.get(word, 0) + 1

    sorted_word_to_cnt = sorted(counter.items(), key=itemgetter(1), reverse=True)
    sorted_words = [x[0] for x in sorted_word_to_cnt]
    words = ['<PAD>', '<UNK>'] + sorted_words
    if len(words) > vocab_size:
        words = words[:vocab_size]
    open(vocab_dir, mode='w', encoding='utf-8').write('\n'.join(words))
    """
    counter = Counter(all_data)
    print("All vocab size is {}.".format(len(counter)))
    count_pairs = counter.most_common(vocab_size - 2)
    # print(count_pairs)
    print("Select vocab size is {}.".format(vocab_size))
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>', '<UNK>'] + list(words)
    open(vocab_dir, mode='w', encoding='utf-8').write('\n'.join(words))
    """


def read_vocab(vocab_dir):
    """读取词汇表"""
    # words = open_file(vocab_dir).read().strip().split('\n')
    with open(vocab_dir, 'r', encoding='utf-8') as fp:
        words = [_.strip() for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def process_file(x_text, y, vocab_file, seq_length):
    """get input_x, input_mask, input_y"""
    # x_text, y = load_polarity_data_and_labels(pos_file, neg_file)
    _, word_to_id = read_vocab(vocab_file)
    x_id = []
    mask = []
    for i in range(len(x_text)):
        content_id = []
        for x in x_text[i].split():
            if x in word_to_id:
                content_id.append(word_to_id[x])
            else:
                content_id.append(1)
        x_id.append(content_id)
        mask.append([1] * len(x_text[i].split()))
    x_pad = kr.preprocessing.sequence.pad_sequences(x_id, seq_length, padding='post')
    x_mask = kr.preprocessing.sequence.pad_sequences(mask, seq_length, padding='post')
    y = np.array(y)
    print("padding数据集完成，共有{}条数据".format(len(x_pad)))
    print("打印出该数据集第一个文本：{}，对应的mask为：{}，第一个标签：{}"
          .format(x_pad[0], x_mask[0], y[0]))
    return x_pad, x_mask, y


def batch_iter(x, x_mask, y, batch_size):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size)  # 这样计算能准确计算出batch数
    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    x_mask_shuffle = x_mask[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = (i + 1) * batch_size
        yield x_shuffle[start_id:end_id], x_mask_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


if __name__ == "__main__":
    train_data = "data/cnews/train_data.txt"
    test_data = "data/MR/test.txt"
    a, b = load_sst_data(train_data)
    print(a[:10])
    print(b[:20])
    build_vocab(a, 'cnews_vocab.txt', 900000)
    #vision_length(a)
