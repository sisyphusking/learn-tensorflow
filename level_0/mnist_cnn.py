import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('../data/mnist', one_hot=True)

batch_size = 100
n_batch = mnist.train.num_examples//batch_size


# 初始化权值
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)  # 生成一个截断的正态分布
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


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


x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])


# 转换x的格式为4d向量 [batch, in_height, in_width, in_channels]
# -1：代表的是以后面为准
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 初始化第一个卷积层权值和偏置项
W_conv1 = weight_variable([5, 5, 1, 32])  # 5*5*1的卷积核，总共32个
b_conv1 = weight_variable([32])  # 每个卷积核一个偏置项


# 卷积计算，然后应用relu函数
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 初始化第二个卷积层的权值和偏置
W_conv2 = weight_variable([5, 5, 32, 64])  # 32是特征图个数，也就是通道数，64是这一层的卷积核个数
b_conv2 = bias_variable([64])

# 将h_pool1和W_conv2进行卷积，加上偏置项，应用relu函数
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 28*28*1的图片经过第一次卷积和池化后，变成了14*14*32
# 经过第二次卷积和池化，变成了7*7*64

# 初始化第一个全连接层
W_fc1 = weight_variable([7*7*64, 1024])  # 上一层有7*7*64个神经元，全连接层有1024个神经元
b_fc1 = bias_variable([1024])

# 将最后一个池化层的输出压平，变成一维
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

# 求第一个全连接层的输出
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1)+b_fc1)

# keep_prob用来表示神经元的输出概率,也即是dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 初始化第二个全连接层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

# 计算输出
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2)+b_fc2)

# 交叉熵损失
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=prediction))
# 使用AdamOptimizer优化
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 将结果存放在一个布尔列表中
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.7})

        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
        print("Iter"+str(epoch) + ", Testing Accuracy= "+str(acc))

