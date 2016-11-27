import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import data_loader
import matplotlib.pyplot as plt

class SeriesPredictor:

    def __init__(self, input_dim, seq_size, hidden_dim):
        # Hyperparameters
        self.input_dim = input_dim
        self.seq_size = seq_size
        self.hidden_dim = hidden_dim

        # Weight variables and input placeholders
        self.W_out = tf.Variable(tf.random_normal([hidden_dim, 1]), name='W_out')
        self.b_out = tf.Variable(tf.random_normal([1]), name='b_out')
        self.x = tf.placeholder(tf.float32, [None, seq_size, input_dim])
        self.y = tf.placeholder(tf.float32, [None, seq_size])

        # Cost optimizer
        self.cost = tf.reduce_mean(tf.square(self.model() - self.y))
        self.train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.cost)

        # Auxiliary ops
        self.saver = tf.train.Saver()

    def model(self):
        """
        :param x: inputs of size [T, batch_size, input_size]
        :param W: matrix of fully-connected output layer weights
        :param b: vector of fully-connected output layer biases
        """
        cell = rnn_cell.BasicLSTMCell(self.hidden_dim)
        outputs, states = rnn.dynamic_rnn(cell, self.x, dtype=tf.float32)
        num_examples = tf.shape(self.x)[0]
        W_repeated = tf.tile(tf.expand_dims(self.W_out, 0), [num_examples, 1, 1])
        out = tf.batch_matmul(outputs, W_repeated) + self.b_out
        out = tf.squeeze(out)
        return out

    def train(self, train_x, train_y):
        with tf.Session() as sess:
            tf.get_variable_scope().reuse_variables()
            sess.run(tf.initialize_all_variables())
            for i in range(10000):
                _, mse = sess.run([self.train_op, self.cost], feed_dict={self.x: train_x, self.y: train_y})
                if i % 100 == 0:
                    print(i, mse)
            save_path = self.saver.save(sess, 'model.ckpt')
            print('Model saved to {}'.format(save_path))

    def test(self, sess, test_x):
        tf.get_variable_scope().reuse_variables()
        self.saver.restore(sess, 'model.ckpt')
        output = sess.run(self.model(), feed_dict={self.x: test_x})
        return output


def plot_results(train_x, predictions):
    print('plotting')
    plt.figure()
    print('train', list(range(len(train_x))))
    num_train = len(train_x)
    plt.plot(list(range(num_train)), train_x, color='b')
    print('predicted', list(range(num_train, num_train + len(predictions))))
    plt.plot(list(range(num_train, num_train + len(predictions))), predictions, color='r')
    plt.show()


if __name__ == '__main__':
    seq_size = 10
    predictor = SeriesPredictor(input_dim=1, seq_size=seq_size, hidden_dim=100)
    data = data_loader.load_series('international-airline-passengers.csv')
    train_data, test_data = data_loader.split_data(data)
    print('train_data', np.shape(train_data))
    print('test_data', np.shape(test_data))

    train_x = []
    train_y = []
    for i in range(len(train_data) - seq_size - 1):
        train_x.append(np.expand_dims(train_data[i:i+seq_size], axis=1).tolist())
        train_y.append(train_data[i+1:i+seq_size+1])
    print('train_x', np.shape(train_x))
    print('train_y', np.shape(train_y))

    predictor.train(train_x, train_y)

    prev_seq = train_x[-1]
    predicted_vals = []
    with tf.Session() as sess:
        for i in range(20):
            next_seq = predictor.test(sess, [prev_seq])
            print('next_seq', next_seq, next_seq[-1])
            predicted_vals.append(next_seq[-1])
            prev_seq = np.expand_dims(next_seq, axis=1)

    plot_results(train_data, predicted_vals)
