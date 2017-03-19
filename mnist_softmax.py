import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

X = tf.placeholder(tf.float32, [None, 28 * 28])
W = tf.Variable(tf.zeros([28 * 28, 10]))
b = tf.Variable(tf.zeros([10]))
Y = tf.nn.softmax(tf.matmul(X, W) + b)

Y_ = tf.placeholder("float", [None, 10])
cross_entropy = - tf.reduce_sum(Y_ * tf.log(Y))

learning_rate = 0.01
epoches = 2000
batch_size = 100
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for epoch in range(epoches):
    xs, ys = mnist.train.next_batch(batch_size)
    _, c = sess.run([train_step, cross_entropy], feed_dict={X : xs, Y_: ys})
    total_batch = int(mnist.train.num_examples / batch_size)
    if epoch % 10 == 0:
        print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c/total_batch))

correct_predict = tf.equal(tf.argmax(Y,1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predict, "float"))
result = sess.run(accuracy, feed_dict={X : mnist.test.images, Y_ : mnist.test.labels})

print("result=",result)#about 0.91