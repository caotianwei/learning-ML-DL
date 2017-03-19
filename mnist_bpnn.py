import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

learning_rate = 0.001
epochs = range(3000)
batch_size = 110

n_input = 28 * 28
n_hidden = 650
n_classes = 10

x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

weights = {'hidden' : tf.Variable(tf.random_normal([n_input, n_hidden])),
           'out' : tf.Variable(tf.random_normal([n_hidden, n_classes]))}

bias = {'hidden' : tf.Variable(tf.random_normal([n_hidden])),
        'out' : tf.Variable(tf.random_normal([n_classes]))}

def multilayer_perceptron(x, weights, bias):
    hidden_layer = tf.matmul(x, weights['hidden']) + bias['hidden']
    hidden_layer = tf.nn.relu(hidden_layer)
    return tf.matmul(hidden_layer, weights['out']) + bias['out']

predict = multilayer_perceptron(x, weights, bias)
cost = tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=y)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in epochs:
        xs, ys = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict= {x : xs, y : ys})

    correct = tf.equal(tf.argmax(predict, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    acc_val = sess.run(accuracy, feed_dict={x : mnist.test.images, y : mnist.test.labels})
    print('accuracy:', acc_val)#about 0.9488