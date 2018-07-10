#coding:utf-8
import tensorflow as tf
import numpy as np
import pickle


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    # sequence_length-最长词汇数
    # num_classes-分类数
    # vocab_size-总词汇数，语料库的词典大小
    # embedding_size-词向量长度
    # filter_sizes-卷积核尺寸3，4，5
    # num_filters-卷积核数量
    # l2_reg_lambda-l2正则化系数
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            #  相当于word2vec中的词向量，准确来讲应该是使用训练好的word2vec向量，如下
            # with open('embeddings.pickle', 'rb') as f:
            #     embeddings = pickle.load(f)
            # self.W = tf.Variable(embeddings, name='W')  # 如果是引用网上很成熟的词向量，那么这里可以不训练，直接设置成false

            # 这是随机初始化的
            self.W = tf.Variable(
               tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
               name="W")
            # [batch_size, sequence_length, embedding_size]
            # 选取张量W里索引为input_x的元素
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            # 添加一个维度，[batch_size, sequence_length, embedding_size, 1] ，这里就相当于batch_size个图片，1相当于黑白图片色阶
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        
        pooled_outputs = []
        # 分别取3、4、5个长度的卷积核进行卷积操作，卷积核的列宽就是词向量的宽度（embedding_size）
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer ，num_filters：卷积核个数 ，前面三个参数是卷积核尺寸、维度、色阶
                # 这里的filter_size不是3*3这种，而是3*embedding_size，卷积窗口其实是只往下移动，见docs中图示
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                # h的size为: (?, sequence_length - filter_size + 1, 1, 128)
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                # 这里卷积层经过卷积、relu后，句子的长度已经变成sequence_length - filter_size + 1
                # 池化层的kernal是长方形的，将整个feature_map映射到一个点上. 一步到位, 只有一个池化层
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],  # 第二个参数代表窗口长度，这里是句子总长
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)  # 将池化层保留下来

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)  # 总共有128*3个卷积核
        self.h_pool = tf.concat(pooled_outputs, 3)  # 3个池化层进行拼接， 变成(?, 1, 1, 384)
        # 把池化层输出变成一维向量，拉平后变成(?, 384)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        
        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.softmax(tf.nn.xw_plus_b(self.h_drop, W, b, name="scores"))
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
