import tensorflow as tf


a = tf.constant(5, name="input_a")
b = tf.constant(3, name="input_b")
c = tf.multiply(a, b, name="mul_c")
d = tf.add(a, b, name="add_d")
e = tf.add(c, d, name="add_e")

with tf.Session() as sess:
    print(sess.run(e))  # output => 23
    writer = tf.summary.FileWriter("../data/graph/hello", sess.graph)

# cd data/graph/
# tensorboard --logdir="./hello"

