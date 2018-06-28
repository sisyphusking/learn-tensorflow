import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

num = 100
xs = np.linspace(-3, 3, num)
ys = np.sin(xs) + np.random.uniform(-0.5, 0.5, num)

# placeholder：占位符
X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

# name命名要符合变量命名规范
W = tf.Variable(tf.random_normal([1]), name='weight')
W_2 = tf.Variable(tf.random_normal([1]), name='weight_2')
W_3 = tf.Variable(tf.random_normal([1]), name='weight_3')
b = tf.Variable(tf.random_normal([1]), name='b')


Y_pred = tf.add(tf.multiply(X, W), b)
Y_pred = tf.add(Y_pred, tf.multiply(tf.pow(X, 2), W_2))
Y_pred = tf.add(Y_pred, tf.multiply(tf.pow(X, 3), W_3))

n_sample = xs.shape[0]
# reduce_sum：求和，起到降维的作用
loss = tf.reduce_sum(tf.square(Y-Y_pred, name='loss'))/n_sample


learn_rate = 0.03
optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss)


with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('../data/graph/Polynomial-regression', sess.graph)
    for i in range(500):
        total_loss = 0
        for x, y in zip(xs, ys):
            _, l = sess.run([optimizer, loss], feed_dict={X:x, Y:y})
            total_loss += l
        if i%10 == 0:
            print('epoch {0}: {1}'.format(i, total_loss))
    writer.close()
    W, W_2, W_3, b = sess.run([W, W_2, W_3, b])


print('trained W: {}'.format(W))
print('trained W_2: {}'.format(W_2))
print('trained W_3: {}'.format(W_3))
print('trained b: {}'.format(b))
plt.plot(xs, ys, 'bo', label='real data')
plt.plot(xs, xs*W+np.power(xs, 2)*W_2+np.power(xs, 3)*W_3+b, 'r', label='predict data')
plt.legend()
plt.title('Polynomial Regression')
plt.show()

