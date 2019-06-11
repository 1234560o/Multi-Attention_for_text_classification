# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author:zwj

import numpy as np


def load_glove(vocab_path, vec_path):
    word2vec = {}
    words = []
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for content in f: 
            if content.strip() != '':
                words.append(content.strip())

    with open(vec_path, 'r', encoding='utf-8') as g:
        for line in g:
            if line.strip() != '':
                tmp = line.strip().split(" ", 1)
                word2vec[tmp[0]] = [float(vec) for vec in tmp[1].split()]
            embedding_dim = len(tmp[1].split())
    embedding = []
    count = 0
    for word in words:
        if word in word2vec.keys():
            embedding.append(word2vec[word])
            count += 1
        else:
            embedding.append(list(np.random.normal(size=embedding_dim)))
    print("共有{}/{}个词使用了预训练词向量.".format(count, len(words)))
    print("预训练的embedding table维度大小是[{} , {}]".format(len(embedding), embedding_dim))
    return embedding


if __name__ == "__main__":
    vocab_path = 'data/rt-polaritydata/vocab111.txt'
    vec_path = 'data/glove.6B.50d.txt'
    emb = load_glove(vocab_path, vec_path)
    print(len(emb), len(emb[0]))
