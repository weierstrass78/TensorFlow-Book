import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell

hidden_dim = 10
input_dim = 1
batch_size = 4

W_out = tf.Variable(tf.random_normal([hidden_dim, 1]))
b_out = tf.Variable(tf.random_normal([1]))
x = tf.placeholder(tf.float32, [None, batch_size, input_dim])
y = tf.placeholder(tf.float32, [None, batch_size])


def model(x, W, b):
    """
    :param x: inputs of size [T, batch_size, input_size]
    :param W: matrix of fully-connected output layer weights
    :param b: vector of fully-connected output layer biases
    """
    cell = rnn_cell.BasicLSTMCell(hidden_dim)
    outputs, states = rnn.dynamic_rnn(cell, x, dtype=tf.float32)
    num_examples = tf.shape(x)[0]
    W_repeated = tf.tile(tf.expand_dims(W, 0), [num_examples, 1, 1])
    out = tf.batch_matmul(outputs, W_repeated) + b
    out = tf.squeeze(out)
    return out

model_op = model(x, W_out, b_out)
cost = tf.reduce_mean(tf.square(model_op - y))
train_op = tf.train.AdamOptimizer().minimize(cost)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    batch1 = [[1], [2], [5], [6]]
    y1 = [1, 3, 7, 11]
    batch2 = [[5], [7], [7], [8]]
    y2 = [5, 12, 14, 15]
    batch3 = [[3], [4], [5], [7]]
    y3 = [3, 7, 9, 12]
    data = [batch1, batch2, batch3]
    response = [y1, y2, y3]
    model_val = sess.run(model_op, feed_dict={x: data, y: response})
    print(np.shape(model_val))
    print(model_val)

    mse = sess.run(cost, feed_dict={x: data, y: response})
    print('original mse', mse)

    for i in range(1000):
        _, mse = sess.run([train_op, cost], feed_dict={x: data, y: response})
        print(i, mse)

    test_data = [[[1], [2], [3], [4]],
                 [[4], [5], [6], [7]]]
    test_val = sess.run(model_op, feed_dict={x: test_data})
    print(test_val)