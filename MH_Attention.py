# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author:zwj

import tensorflow as tf
import math

def gelu(input_tensor):
  """Gaussian Error Linear Unit.

  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415

  Args:
    input_tensor: float Tensor to perform activation.

  Returns:
    `input_tensor` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
  return input_tensor * cdf


class Text_attention2(object):
    def __init__(self,
                 glove_vec,
                 seq_length,
                 num_classes,
                 embedding_size,
                 vocab_size,
                 batch_size,
                 hidden_size,
                 num_attention_heads,
                 num_hidden_layers,
                 sent_attention_size,
                 initializer_range=0.02,
                 base_learning_rate=0.1,
                 decay_rate=0.95
                 ):
        self.glove_vec = glove_vec
        self.seq_length = seq_length
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.initializer_range = initializer_range
        self.base_learning_rate = base_learning_rate
        self.decay_rate = decay_rate

        self.input_x = tf.placeholder(tf.int32, [self.batch_size, self.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.int64, [self.batch_size], name='input_y')
        self.hidden_dropout_prob = tf.placeholder(tf.float32, name="hidden_dropout_prob")
        self.attention_probs_dropout_prob = tf.placeholder(tf.float32, name="attention_probs_dropout_prob")
        self.input_mask = tf.placeholder(tf.int32, [self.batch_size, self.seq_length])

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (self.hidden_size, self.num_attention_heads))

        with tf.device('/cpu:0'):
            embedding_inputs = self.embedding_postprocessor()

        attention_mask = self.create_attention_mask_from_input_mask()
        prev_output = tf.reshape(embedding_inputs, [self.batch_size * self.seq_length, self.embedding_size])
        all_layer_outputs = []
        for layer_idx in range(self.num_hidden_layers):
            with tf.variable_scope("layer_%d" % layer_idx):
                layer_input = prev_output
                with tf.variable_scope("attention"):
                    with tf.variable_scope("self"):
                        attention_output = self.self_attention_layer(embedding_inputs=layer_input,
                                                                     attention_mask=attention_mask)
                    with tf.variable_scope("output"):
                        attention_output = tf.layers.dense(
                            attention_output,
                            self.hidden_size,
                            kernel_initializer=create_initializer(self.initializer_range))
                        attention_output = dropout(attention_output, self.hidden_dropout_prob)
                        # attention_output = layer_norm(attention_output + layer_input)

                        # prev_output = layer_output
                        prev_output = attention_output
                        all_layer_outputs.append(attention_output)
        self.final_outputs = []
        for layer_output in all_layer_outputs:
            final_output = tf.reshape(layer_output, [self.batch_size, self.seq_length, self.hidden_size])
            self.final_outputs.append(final_output)

        with tf.variable_scope("pooler"):
            # self.pooled_out = tf.transpose(self.final_outputs[-1], [1, 0, 2])
            average_num = tf.reshape(tf.cast(tf.reduce_sum(self.input_mask, axis=1), tf.float32), [self.batch_size, 1])
            fc_mask = tf.cast(tf.reshape(self.input_mask, [self.batch_size, 1, self.seq_length]), tf.float32)
            # get [B, 1, hidden_size] to [B, hidden_size]
            self.pooled_out = tf.reshape(tf.matmul(fc_mask, self.final_outputs[-1]), [self.batch_size, self.hidden_size])
            self.pooled_out  = self.pooled_out / average_num
            print("self.pooled_out维度是{}".format(self.pooled_out.shape))
            fc1 = tf.layers.dense(self.pooled_out, self.hidden_size, kernel_initializer=create_initializer(initializer_range))
            self.scores = tf.layers.dense(fc1, self.num_classes, kernel_initializer=create_initializer(initializer_range))
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        with tf.variable_scope("loss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(
                self.base_learning_rate,
                global_step,
                100,
                self.decay_rate)
            # 优化器
            self.optim = tf.train.AdamOptimizer(
                learning_rate=learning_rate).minimize(self.loss, global_step=global_step)

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, self.input_y)
            self.acc = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    # 得到词向量（包括位置词向量）
    def embedding_postprocessor(self,
                                word_embedding_name="word_embeddings",
                                use_position_embeddings=True,
                                position_embedding_name="position_embeddings",
                                max_position_embeddings=512):
        if self.glove_vec:
            embedding_table = tf.get_variable(
                'embedding', initializer=tf.constant(self.glove_vec, dtype=tf.float32))
            self.embedding_size = len(self.glove_vec[0])
        else:
            embedding_table = tf.get_variable(
                name=word_embedding_name,
                shape=[self.vocab_size, self.embedding_size],
                initializer=create_initializer(self.initializer_range))
        embedding_inputs = tf.nn.embedding_lookup(embedding_table, self.input_x)
        if use_position_embeddings:
            assert_op = tf.assert_less_equal(self.seq_length, max_position_embeddings)
            with tf.control_dependencies([assert_op]):
                full_position_embeddings = tf.get_variable(
                    name=position_embedding_name,
                    shape=[max_position_embeddings, self.embedding_size],
                    initializer=create_initializer(self.initializer_range))

                position_embeddings = tf.slice(full_position_embeddings, [0, 0], [self.seq_length, -1])
                num_dims = len(embedding_inputs.shape.as_list())

                # Only the last two dimensions are relevant (`seq_length` and `embedding_size`), so
                # we broadcast among the first dimensions, which is typically just
                # the batch size.
                position_broadcast_shape = []
                for _ in range(num_dims - 2):
                    position_broadcast_shape.append(1)
                position_broadcast_shape.extend([self.seq_length, self.embedding_size])
                # 将position_embeddings变成和output相同的维度以便相加
                position_embeddings = tf.reshape(position_embeddings,
                                                 position_broadcast_shape)
                embedding_inputs += position_embeddings

                # LN和droupout处理，这里的droupout为1-droupout_prob
                embedding_inputs = layer_norm_and_dropout(embedding_inputs, self.hidden_dropout_prob)
        return embedding_inputs

    def create_attention_mask_from_input_mask(self):
        to_mask = tf.cast(tf.reshape(self.input_mask, [self.batch_size, 1, self.seq_length]), tf.float32)
        broadcast_ones = tf.ones(shape=[self.batch_size, self.seq_length, 1], dtype=tf.float32)
        mask = broadcast_ones * to_mask
        attention_mask = tf.expand_dims(mask, axis=[1])
        return attention_mask

    def self_attention_layer(self,
                             embedding_inputs,
                             attention_mask=None,
                             query_act=None,
                             key_act=None,
                             value_act=None):

        def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                                 seq_length, width):
            output_tensor = tf.reshape(
                input_tensor, [batch_size, seq_length, num_attention_heads, width])

            # 第二个维度变成num_attention_heads，相当于切割
            output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
            return output_tensor

        from_tensor_2d = tf.reshape(embedding_inputs, [self.batch_size * self.seq_length, -1])
        to_tensor_2d = tf.reshape(embedding_inputs, [self.batch_size * self.seq_length, -1])

        query_layer = tf.layers.dense(
            from_tensor_2d,
            self.hidden_size,
            activation=query_act,
            name="query",
            kernel_initializer=create_initializer(self.initializer_range))

        key_layer = tf.layers.dense(
            to_tensor_2d,
            self.hidden_size,
            activation=key_act,
            name="key",
            kernel_initializer=create_initializer(self.initializer_range))

        value_layer = tf.layers.dense(
            to_tensor_2d,
            self.hidden_size,
            activation=value_act,
            name="value",
            kernel_initializer=create_initializer(self.initializer_range))

        attention_head_size = int(self.hidden_size / self.num_attention_heads)
        query_layer = transpose_for_scores(query_layer, self.batch_size,
                                           self.num_attention_heads, self.seq_length,
                                           attention_head_size)
        key_layer = transpose_for_scores(key_layer, self.batch_size, self.num_attention_heads,
                                         self.seq_length, attention_head_size)
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        attention_scores = tf.multiply(attention_scores,
                                       1.0 / math.sqrt(float(attention_head_size)))

        if attention_mask is not None:
            adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0
            attention_scores += adder

        attention_probs = tf.nn.softmax(attention_scores)
        attention_probs = dropout(attention_probs, self.attention_probs_dropout_prob)
        value_layer = transpose_for_scores(value_layer, self.batch_size, self.num_attention_heads,
                                         self.seq_length, attention_head_size)
        context_layer = tf.matmul(attention_probs, value_layer)
        context_layer = tf.transpose(context_layer, [0, 2, 1, 3])
        context_layer = tf.reshape(
            context_layer,
            [self.batch_size * self.seq_length, self.num_attention_heads * attention_head_size])
        return context_layer


"""
    Original taken from https://github.com/google-research/bert/blob/master/modeling.py
"""

def dropout(input_tensor, dropout_prob):
  """Perform dropout.

  Args:
    input_tensor: float Tensor.
    dropout_prob: Python float. The probability of dropping out a value (NOT of
      *keeping* a dimension as in `tf.nn.dropout`).

  Returns:
    A version of `input_tensor` with dropout applied.
  """
  if dropout_prob is None or dropout_prob == 0.0:
    return input_tensor

  output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
  return output


def layer_norm(input_tensor, name=None):
  """Run layer normalization on the last dimension of the tensor."""
  return tf.contrib.layers.layer_norm(
      inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)


def layer_norm_and_dropout(input_tensor, dropout_prob, name=None):
  """Runs layer normalization followed by dropout."""
  output_tensor = layer_norm(input_tensor, name)
  output_tensor = dropout(output_tensor, dropout_prob)
  return output_tensor


def create_initializer(initializer_range=0.02):
  """Creates a `truncated_normal_initializer` with the given range."""
  return tf.truncated_normal_initializer(stddev=initializer_range)


# test
if __name__ == "__main__":
    model = Text_attention(
        seq_length=5,
        num_classes=2,
        vocab_size=5,
        batch_size=2,
        hidden_size=6,
        num_attention_heads=2,
        num_hidden_layers=2,
        intermediate_size=15)
    input_x = [[1,2,2,0,0], [2,4,3,2,0]]
    input_y = [1, 0]
    input_mask = [[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        a, b = sess.run([model.accuracy, model.loss], feed_dict={model.input_x: input_x,
                                                                   model.input_y: input_y,
                                                                  model.input_mask: input_mask,
                                                                 model.attention_probs_dropout_prob: 0.0,
                                                                model.hidden_dropout_prob: 0.0})
        print(a)
        print(b)
