# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author:zwj

import time
from datetime import timedelta

from train import get_time_dif, evaluate, feed_data, FLAGS
import numpy as np
import tensorflow as tf
from sklearn import metrics
from data_helpers import *
from MH_Attention import Text_attention2
from MHFF_Attention import Text_attention
from load_pre_vec import load_glove
from sklearn.metrics import roc_curve, auc
plt.rcParams['font.sans-serif'] = ['SimHei']   # 为了显示中文字符
plt.rcParams['axes.unicode_minus'] = False

save_path = 'checkpoints/best_validation'


def eval_MHFF(x_dev, mask_dev, y_dev):
    if FLAGS.use_pre_vec:
        glove_vec = load_glove(FLAGS.base_dir + FLAGS.vocab_file, FLAGS.vec_file)
    else:
        glove_vec = None
    model = Text_attention(glove_vec,
                           FLAGS.seq_length,
                           FLAGS.num_classes,
                           FLAGS.embedding_size,
                           FLAGS.vocab_size,
                           FLAGS.batch_size,
                           FLAGS.hidden_size,
                           FLAGS.num_attention_heads,
                           FLAGS.num_hidden_layers,
                           FLAGS.sent_attention_size,
                           FLAGS.initializer_range,
                           FLAGS.base_learning_rate,
                           FLAGS.decay_rate)
    print("Loading test data...")
    start_time = time.time()
    # x_test, y_test = load_polarity_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
    # x_test, mask, y = process_file(x_test, y_test, FLAGS.vocab_file, FLAGS.seq_length)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型

    print('Testing...')
    loss_test, acc_test = evaluate(session, model, x_dev, mask_dev, y_dev, FLAGS.batch_size)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))
    """
    data_len = len(x_dev)
    num_batch = int((data_len - 1) // FLAGS.batch_size)

    y_pred = np.zeros(shape=num_batch * FLAGS.batch_size, dtype=np.int32)  # 保存预测结果
    y_scores = np.zeros(shape=[num_batch * FLAGS.batch_size, FLAGS.num_classes], dtype=np.float64)
    attention_score = np.zeros(shape=[num_batch * FLAGS.batch_size, FLAGS.seq_length],
                               dtype=np.float64)
    for i in range(num_batch):  # 逐批次处理
        start_id = i * FLAGS.batch_size
        end_id = (i + 1) * FLAGS.batch_size
        feed_dict = feed_data(model,
                              x_dev[start_id: end_id],
                              mask_dev[start_id: end_id],
                              y_dev[start_id: end_id],
                              0,
                              0)
        y_pred[start_id:end_id], y_scores[start_id:end_id, :], attention_score[start_id:end_id, :] = session.run(
            [model.predictions, model.scores, model.attention_coeff], feed_dict=feed_dict)
    session.close()

    # 评估
    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_dev[: num_batch * FLAGS.batch_size], y_pred,
                                        target_names=['negative', 'positive'],
                                        digits=3))

    # 混淆矩阵
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_dev[: num_batch * FLAGS.batch_size], y_pred)
    print(cm)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    """

def eval_MH(x_dev, mask_dev, y_dev):
    if FLAGS.use_pre_vec:
        glove_vec = load_glove(FLAGS.vocab_file, FLAGS.vec_file)
    else:
        glove_vec = None
    model = Text_attention2(glove_vec,
                           FLAGS.seq_length,
                           FLAGS.num_classes,
                           FLAGS.embedding_size,
                           FLAGS.vocab_size,
                           FLAGS.batch_size,
                           FLAGS.hidden_size,
                           FLAGS.num_attention_heads,
                           FLAGS.num_hidden_layers,
                           FLAGS.sent_attention_size,
                           FLAGS.initializer_range,
                           FLAGS.base_learning_rate,
                           FLAGS.decay_rate)
    print("Loading test data...")
    start_time = time.time()
    # x_test, y_test = load_polarity_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
    # x_test, mask, y = process_file(x_test, y_test, FLAGS.vocab_file, FLAGS.seq_length)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型

    print('Testing...')
    loss_test, acc_test = evaluate(session, model, x_dev, mask_dev, y_dev, FLAGS.batch_size)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))

    """
    data_len = len(x_dev)
    num_batch = int((data_len - 1) // FLAGS.batch_size)

    y_pred = np.zeros(shape=num_batch * FLAGS.batch_size, dtype=np.int32)  # 保存预测结果
    y_scores = np.zeros(shape=[num_batch * FLAGS.batch_size, FLAGS.num_classes], dtype=np.float64)
    # attention_score = np.zeros(shape=[num_batch * FLAGS.batch_size, FLAGS.seq_length],
    #                         dtype=np.float64)
    for i in range(num_batch):  # 逐批次处理
        start_id = i * FLAGS.batch_size
        end_id = (i + 1) * FLAGS.batch_size
        feed_dict = feed_data(model,
                              x_dev[start_id: end_id],
                              mask_dev[start_id: end_id],
                              y_dev[start_id: end_id],
                              0,
                              0)
        y_pred[start_id:end_id], y_scores[start_id:end_id, :] = session.run(
            [model.predictions, model.scores], feed_dict=feed_dict)
    session.close()

    # 评估
    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_dev[: num_batch * FLAGS.batch_size], y_pred,
                                        target_names=['negative', 'positive'],
                                        digits=3))

    # 混淆矩阵
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_dev[: num_batch * FLAGS.batch_size], y_pred)
    print(cm)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    """


if __name__ == "__main__":
    x_test, y_test = load_data(FLAGS.base_dir + 'val_data.txt')
    x_test, mask_test, y_test = process_file(x_test, y_test, FLAGS.base_dir + 'vocab.txt', FLAGS.seq_length)
    x_test = np.array(x_test)
    mask_test = np.array(mask_test)
    y_test = np.array(y_test)
    eval_MHFF(x_test, mask_test, y_test)
