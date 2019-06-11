# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author:zwj

import os
import sys
from datetime import timedelta
import numpy as np
import tensorflow as tf
import time
from data_helpers import *
from MH_Attention import Text_attention2
from MHFF_Attention import Text_attention
from load_pre_vec import load_glove
# from load_imdb import process_imdb
import os

# Parameters
# ==================================================

# Data load params
tf.flags.DEFINE_string("base_dir", "data/MR/", "Data source")
tf.flags.DEFINE_float("dev_sample_percentage", 0.1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")
tf.flags.DEFINE_string("vocab_file", "vocab.txt", "Data source for the vocab")
tf.flags.DEFINE_string("vec_file", "./data/glove.6B.50d.txt", "Data source for pre vec")
tf.flags.DEFINE_integer("seed", 12345, "Random number seed")

# Model Hyper_parameters
tf.flags.DEFINE_boolean("use_pre_vec", True, "Whether use pre vec")
tf.flags.DEFINE_integer("seq_length", 50, "Length of one sentence (default: 64)")
tf.flags.DEFINE_integer("num_classes", 2, "Number of categories (default: 2)")
tf.flags.DEFINE_integer("vocab_size", 14000, "Size of vocab (default: 14000)")
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("embedding_size", 32, "Dimensionality of character embedding(default: 128)")
tf.flags.DEFINE_integer("num_attention_heads", 2, "Number of Multi-Head Attention's head (default: 2)")
tf.flags.DEFINE_integer("num_hidden_layers", 1, "Number of attention layer (default: 1)")
tf.flags.DEFINE_integer("hidden_size", 32, "Dimensionality of hidden_size(default: 128)")
tf.flags.DEFINE_integer("num_layers", 1, "Nums of layers (default: 1)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.8,
                      "The dropout probability for all fully connected layers (default: 0.8)")
tf.flags.DEFINE_float("base_learning_rate", 0.001, "Init learning_rate")
tf.flags.DEFINE_float("decay_rate", 0.92, "Decay_rate of learning_rate")
tf.flags.DEFINE_float("hidden_dropout_prob", 0.1,
                      "The dropout probability for all fully connected layers (default: 0.1)")
tf.flags.DEFINE_float("attention_probs_dropout_prob", 0.1,
                      "The dropout ratio for the attention probabilities (default: 0.1)")
tf.flags.DEFINE_float("initializer_range", 0.02, "(default: 0.02)")
tf.flags.DEFINE_integer("sent_attention_size", 16, " (default: 16)")

# Training parameters
tf.flags.DEFINE_integer("num_epochs", 20, "Number of training epochs (default: 20)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_string("save_dir", 'checkpoints/', "Directory of checkpoints to store")
tf.flags.DEFINE_string("tensorboard_dir", "tensorboard/", "Directory of tensorboard")
tf.flags.DEFINE_integer("print_per_batch", 50, "How many steps to input results of train and dev data")
tf.flags.DEFINE_integer("require_improvement", 2000, "How many steps  when the dev not improve to stops")
tf.flags.DEFINE_integer("save_per_batch", 5, "How many steps to write to tensorboard")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS


# get train_data, dev_data
def pre_process():
    print("Load data...")
    x_text, y = load_data(FLAGS.base_dir + 'train_data.txt')
    if not os.path.exists(FLAGS.base_dir + 'vocab.txt'):
        print("Build vocab...")
        build_vocab(x_text, FLAGS.base_dir + 'vocab.txt', FLAGS.vocab_size)
    x_train, mask_train, y_train = process_file(x_text, y, FLAGS.base_dir + "vocab.txt", FLAGS.seq_length)

    x_dev1, y_dev1 = load_data(FLAGS.base_dir + 'val_data.txt')
    x_dev, mask_dev, y_dev = process_file(x_dev1, y_dev1, FLAGS.base_dir + 'vocab.txt', FLAGS.seq_length)

    if FLAGS.use_pre_vec:
        glove_vec = load_glove(FLAGS.base_dir + 'vocab.txt', FLAGS.vec_file)
    else:
        glove_vec = None
    return x_train, mask_train, y_train, x_dev, mask_dev, y_dev, glove_vec


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def feed_data(model, x_batch, mask_batch, y_batch, hidden_dropout_prob, attention_probs_dropout_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_mask: mask_batch,
        model.input_y: y_batch,
        model.hidden_dropout_prob: hidden_dropout_prob,
        model.attention_probs_dropout_prob: attention_probs_dropout_prob
    }
    return feed_dict


def evaluate(sess, model, x_, mask_, y_, batch_size):
    """评估在某一数据上的准确率和损失"""
    data_len = len(x_)
    batch_eval = batch_iter(x_, mask_, y_, batch_size)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, mask_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(model, x_batch, mask_batch, y_batch, 0.0, 0.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len


def train():

    print("Loading training and validation data...")

    x_train, mask_train, y_train, x_dev, mask_dev, y_dev, glove_vec= pre_process()

    with tf.Graph().as_default():
        # 在没有GPU时，自动将计算任务转移到CPU上运行，并通过日志验证放置情况
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        # model = Text_attention(glove_vec,
          #                     FLAGS.seq_length,
           #                    FLAGS.num_classes,
            #                   FLAGS.embedding_size,
             #                  FLAGS.vocab_size,
              #                 FLAGS.batch_size,
               #                FLAGS.hidden_size,
                #               FLAGS.num_attention_heads,
                 #              FLAGS.num_hidden_layers,
                  #             FLAGS.initializer_range,
                   #            FLAGS.base_learning_rate,
                    #           FLAGS.decay_rate)
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

        print("Configuring TensorBoard and Saver...")
        # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
        # tensorboard_path = FLAGS.tensorboard_dir + FLAGS.dataset
        # MHFF指modeling2
        # tensorboard_path = FLAGS.tensorboard_dir
        if not os.path.exists(FLAGS.tensorboard_dir):
            os.makedirs(FLAGS.tensorboard_dir)

        tf.summary.scalar("loss", model.loss)
        tf.summary.scalar("accuracy", model.acc)
        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter(FLAGS.tensorboard_dir)

        # 配置 Saver
        saver = tf.train.Saver()
        # checkpoint_path = FLAGS.save_dir + FLAGS.dataset
        checkpoint_path = FLAGS.save_dir
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        save_path = checkpoint_path + '/best_validation'  # 最佳验证结果保存路径

        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Initialize all variables
            sess.run(tf.global_variables_initializer())
            writer.add_graph(sess.graph)

            print('Training and evaluating...')
            start_time = time.time()
            total_batch = 0  # 总批次
            best_acc_val = 0.0  # 最佳验证集准确率
            last_improved = 0  # 记录上一次提升批次

            flag = False
            for epoch in range(FLAGS.num_epochs):
                print('Epoch:', epoch + 1)
                batch_train = batch_iter(x_train, mask_train, y_train, FLAGS.batch_size)
                for x_batch, mask_batch, y_batch in batch_train:
                    feed_dict = feed_data(model, x_batch, mask_batch, y_batch,
                                          FLAGS.hidden_dropout_prob,
                                          FLAGS.attention_probs_dropout_prob)

                    if total_batch % FLAGS.save_per_batch == 0:
                        # 每多少轮次将训练结果写入tensorboard scalar
                        s = sess.run(merged_summary, feed_dict=feed_dict)
                        writer.add_summary(s, total_batch)

                    if total_batch % FLAGS.print_per_batch == 0:
                        # 每多少轮次输出在训练集和验证集上的性能
                        feed_dict[model.hidden_dropout_prob] = 0.0
                        feed_dict[model.attention_probs_dropout_prob] = 0.0
                        loss_train, acc_train = sess.run([model.loss, model.acc], feed_dict=feed_dict)
                        loss_val, acc_val = evaluate(sess, model, x_dev, mask_dev, y_dev, FLAGS.batch_size)  # todo

                        if acc_val > best_acc_val:
                            # 保存最好结果
                            best_acc_val = acc_val
                            last_improved = total_batch
                            saver.save(sess=sess, save_path=save_path)
                            improved_str = '*'
                        else:
                            improved_str = ''

                        time_dif = get_time_dif(start_time)
                        msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                              + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                        print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

                    feed_dict[model.hidden_dropout_prob] = FLAGS.hidden_dropout_prob
                    feed_dict[model.attention_probs_dropout_prob] = FLAGS.attention_probs_dropout_prob
                    sess.run(model.optim, feed_dict=feed_dict)  # 运行优化
                    total_batch += 1

                    if total_batch - last_improved > FLAGS.require_improvement:
                        # 验证集正确率长期不提升，提前结束训练
                        print("No optimization for a long time, auto-stopping...")
                        flag = True
                        break  # 跳出循环
                if flag:  # 同上
                    break
            writer.close()


if __name__ == "__main__":
    train()
