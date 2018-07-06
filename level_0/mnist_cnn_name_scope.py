import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('../data/mnist', one_hot=True)

batch_size = 100
n_batch = mnist.train.num_examples//batch_size


def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope("stddev"):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


# 初始化权值
def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)  # 生成一个截断的正态分布
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def conv2d(x, W):
    """
    :param x:  input tensor of shape [batch, in_height,in_width, in_channels]
    :param W:  kernel tensor of shape [filter_height, filter_width, in_channels, out_channels]
               stride[0] must be equal to stride[3], the value is 1
               step[1]: step length of x axis
               step[2]: step length of y axis
               padding: 'SAME' model will fill with 0, 'VALID' will not.
    :return:
    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


def max_pool_2x2(x):
    # ksize: [1,x,y,1]， x*y: 窗口大小, 第1个和第4个元素必须为1
    # strides: 第1个和第4个元素必须为1，第2个和第3个元素代表窗口移动的步长
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y = tf.placeholder(tf.float32, [None, 10], name='y-input')
    with tf.name_scope('x_image'):
        x_image = tf.reshape(x, [-1, 28, 28, 1], name='x_image')


with tf.name_scope('Conv1'):
    with tf.name_scope("W_conv1"):
        W_conv1 = weight_variable([5, 5, 1, 32], name='W_conv1')
    with tf.name_scope('b_conv1'):
        b_conv1 = bias_variable([32], name='b_conv1')
    with tf.name_scope('h_conv1_relu'):
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1, name='h_conv1_relu')
    with tf.name_scope('h_pool1'):
        h_pool1 = max_pool_2x2(h_conv1)


with tf.name_scope('Conv2'):
    with tf.name_scope("W_conv2"):
        W_conv2 = weight_variable([5, 5, 32, 64], name='W_conv2')
    with tf.name_scope('b_conv2'):
        b_conv2 = bias_variable([64], name='b_conv2')
    with tf.name_scope('h_conv2_relu'):
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2, name='h_conv2_relu')
    with tf.name_scope('h_pool2'):
        h_pool2 = max_pool_2x2(h_conv2)


with tf.name_scope('fc1'):
    with tf.name_scope('W_fc1'):
        W_fc1 = weight_variable([7 * 7 * 64, 1024], name='W_fc1')
    with tf.name_scope('b_fc1'):
        b_fc1 = bias_variable([1024], name='b_fc1')

    with tf.name_scope('h_pool2_flat'):
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    with tf.name_scope('wx_plus_b1'):
        wx_plus_b1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
    with tf.name_scope('relu'):
        h_fc1 = tf.nn.relu(wx_plus_b1)

    with tf.name_scope('keep_prob'):
        keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    with tf.name_scope('h_fc1_drop'):
        h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob, name='h_fc1_drop')


with tf.name_scope('fc2'):
    with tf.name_scope('W_fc2'):
        W_fc2 = weight_variable([1024, 10], name='W_fc2')
    with tf.name_scope('b_fc2'):
        b_fc2 = bias_variable([10], name='b_fc2')
    with tf.name_scope('wx_plus_b2'):
        wx_plus_b2 = tf.matmul(h_fc1_dropout, W_fc2) +b_fc2
    with tf.name_scope('softmax'):
        prediction = tf.nn.softmax(wx_plus_b2)


with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=prediction), name=
                                   'cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter('../data/logs/train', sess.graph)
    test_writer = tf.summary.FileWriter('../data/logs/test', sess.graph)

    for i in range(1001):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5})
        # 记录训练集计算参数
        summary = sess.run(merged, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
        train_writer.add_summary(summary, i)
        # 记录测试集计算参数
        batch_xs, batch_ys = mnist.test.next_batch(batch_size)
        summary = sess.run(merged, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
        train_writer.add_summary(summary, i)

        if i%100 == 0:
            test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
            train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images[:10000],
                                                     y: mnist.train.labels[:10000], keep_prob: 1.0})
            print("Iter"+str(i) + ", Testing Accuracy= "+str(test_acc) + ", Training Accuracy= " + str(train_acc))

