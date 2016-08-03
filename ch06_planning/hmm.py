import tensorflow as tf
import numpy as np

# Num states
K = 2

# Num observations
N = 3

# Observations
Y = [0, 1, 2]

# Prior (K)
prior = tf.constant([0.6, 0.4], dtype=tf.double)

# Transition matrix (K x K)
T = tf.constant([[0.7, 0.3],
                 [0.4, 0.6]], dtype=tf.double)

# Emission matrix (K x N)
B = tf.constant([[0.5, 0.4, 0.1],
                 [0.1, 0.3, 0.6]], dtype=tf.double)

# (K x T)
T1 = tf.Variable(tf.zeros([N, K], dtype=tf.double))
T2 = tf.Variable(tf.zeros([N, K], dtype=tf.double))

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    y0 = Y[0]
    t1_update = tf.mul(B[:, y0], prior)
    t1_update_val = sess.run(t1_update)
    print(np.shape(t1_update_val))
    print(t1_update_val)
    T1_val = tf.scatter_update(T1, [0], [[1., 1.]])
    print(sess.run(T1_val))
