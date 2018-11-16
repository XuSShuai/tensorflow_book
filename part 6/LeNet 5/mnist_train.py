import tensorflow as tf
import mnist_inference
import os
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

# network setting
EPOCH = 30000
BATCH_SIZE = 32
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
MOVEING_AVERAGE_DECAY = 0.99
REGULARAZTION_RATE = 0.001
MODEL_SAVE_PATH = 'model_save'
MODEL_SAVE_NAME = 'conv_connected_model'

def train(mnist):
    x = tf.placeholder(dtype=tf.float32, shape=[None, 
                                                mnist_inference.IMAGE_SIZE,
                                                mnist_inference.IMAGE_SIZE, 
                                                mnist_inference.IMAGE_CHANNELS], 
                       name="x-input")
    y_ = tf.placeholder(dtype=tf.float32, shape=[None, mnist_inference.LABELS], name="y-output")
    
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    y = mnist_inference.inference(x, True, regularizer)
    
    cem_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
    loss = cem_loss + tf.add_n(tf.get_collection("losses"))
    
    global_step = tf.Variable(0, trainable=False)
    
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, 
                                               global_step, 
                                               mnist.train.num_examples/BATCH_SIZE,LEARNING_RATE_DECAY, 
                                               True)
    train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step)
    
    ema = tf.train.ExponentialMovingAverage(MOVEING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    
    train_op = tf.group([train, ema_op])
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), tf.float32))
    
    init_op = tf.global_variables_initializer()
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init_op)
        for i in range(EPOCH):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            xs_reshaped = np.reshape(xs, [-1, 
                                          mnist_inference.IMAGE_SIZE, 
                                          mnist_inference.IMAGE_SIZE, 
                                          mnist_inference.IMAGE_CHANNELS])
            _, step = sess.run([train_op, global_step], feed_dict={x: xs_reshaped, y_: ys})
            if i % 1000 == 0:
                xs = np.reshape(mnist.train.images, [-1, 
                                                     mnist_inference.IMAGE_SIZE, 
                                                     mnist_inference.IMAGE_SIZE, 
                                                     mnist_inference.IMAGE_CHANNELS])
                accuracy_value, loss_value = sess.run([accuracy, loss], feed_dict={x: xs, y_: mnist.train.labels})
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_SAVE_NAME), global_step=step)
                print("after %i steps, the loss is %f, accuracy on training set is %f" % (i, loss_value, accuracy_value))
                
def main(argv=None):
    mnist = input_data.read_data_sets("../../part 5/mnist/", one_hot = True)
    train(mnist)
    
if __name__ == "__main__":
    tf.app.run()