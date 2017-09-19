import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

import bayes

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

def model(input):
    lr = np.exp(input['learning_rate'])
    C = np.exp(input['C'])
    bs = 2 ** input['batch_size']
    epochs = input['epochs'] * 5

    X = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])

    W = tf.get_variable('W', [784, 10],
            initializer=tf.truncated_normal_initializer())
    b = tf.get_variable('b', [10],
            initializer=tf.truncated_normal_initializer())

    pred = tf.nn.softmax(tf.matmul(X, W) + b)

    regularization = tf.nn.l2_loss(W)
    loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))

    loss = tf.reduce_mean(loss + C * regularization)

    optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)

    with tf.Session() as sess:
        # initialize variables
        sess.run(tf.global_variables_initializer())

        for e in range(int(epochs)):
            total_batch = int(mnist.train.num_examples / bs)
            for i in range(total_batch):
                X_batch, y_batch = mnist.train.next_batch(int(bs))
                sess.run(optimizer, feed_dict={X: X_batch, y: y_batch})

        correct = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        v = float(accuracy.eval({X: mnist.test.images, y: mnist.test.labels}))

    # reset graph for next use
    tf.reset_default_graph()
    return v

X_inputs = {
        'learning_rate' : np.log([0.001, 0.003, 0.01, 0.03, 0.1, 0.3]),
        'C' : np.log([0.001, 0.01, 0.1]),
        'batch_size' : range(4, 9),
        'epochs' : range(1, 5)
        }

x_best, acc = bayes.optimize(model, X_inputs, 'm52', 'ei', 10)
print("x_best = {}".format(x_best))
print("accuracy = {}".format(acc))
