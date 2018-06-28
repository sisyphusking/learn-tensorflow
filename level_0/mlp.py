import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../data/mnist", one_hot=True)

X = tf.placeholder(tf.float32, [None, 784], name='X_placeholder')
Y = tf.placeholder(tf.int32, [None, 10], name='Y_placeholder')

# 网络参数
n_hidden_1 = 256  # 第1个隐层
n_hidden_2 = 256  # 第2个隐层
n_input = 784  # MNIST 数据输入(28*28*1=784)
n_classes = 10  # MNIST 总共10个手写数字类别

weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1]), name='W1'),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name='W2'),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]), name='W')
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1]), name='b1'),
    'b2': tf.Variable(tf.random_normal([n_hidden_2]), name='b2'),
    'out': tf.Variable(tf.random_normal([n_classes]), name='bias')
}


def multilayer_perceptron(x, weights, biases):
    # 第1个隐层，使用relu激活函数
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'], name='fc_1')
    layer_1 = tf.nn.relu(layer_1, name='relu_1')
    # 第2个隐层，使用relu激活函数
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'], name='fc_2')
    layer_2 = tf.nn.relu(layer_2, name='relu_2')
    # 输出层
    out_layer = tf.add(tf.matmul(layer_2, weights['out']), biases['out'], name='fc_3')
    return out_layer


pred = multilayer_perceptron(X, weights, biases)

learning_rate = 0.001
# logits：就是神经网络最后一层的输出，如果有batch的话，它的大小就是[batchsize，n_classes]，单样本的话，大小就是n_classes
# labels：实际的标签，大小同上
# 1、先对网络最后一层的输出做一个softmax，求取输出属于某一类的概率
# 2、softmax的输出向量[Y1，Y2,Y3...]和样本的实际标签做一个交叉熵，等价于：-tf.reduce_sum(y_*tf.log(y))
loss_all = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y, name='cross_entropy_loss')
# reduce_mean：求向量的均值
loss = tf.reduce_mean(loss_all, name='avg_loss')
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

init = tf.global_variables_initializer()

# 训练总轮数
training_epochs = 15
# 一批数据大小
batch_size = 128
# 信息展示的频度
display_step = 1

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('../data/graph/mlp', sess.graph)

    # 训练
    for epoch in range(training_epochs):
        avg_loss = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # 遍历所有的batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # 使用optimizer进行优化
            _, l = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y})
            # 求平均的损失
            avg_loss += l / total_batch
        # 每一步都展示信息
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost = {:.9f}".format(avg_loss))
    print("Optimization Finished!")

    # 在测试集上评估
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # 计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))
    writer.close()
