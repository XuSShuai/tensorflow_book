import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference
import mnist_train
import time
import numpy as np

EVAL_INTERVAL_SECS = 60

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(dtype=tf.float32, shape=[None, 
                                                mnist_inference.IMAGE_SIZE,
                                                mnist_inference.IMAGE_SIZE, 
                                                mnist_inference.IMAGE_CHANNELS], name="x-input")
        y_ = tf.placeholder(dtype=tf.float32, shape=[None, mnist_inference.LABELS], name="y-input")

        y = mnist_inference.inference(x, False, None)

        ema = tf.train.ExponentialMovingAverage(mnist_train.MOVEING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)
        
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), tf.float32))
        
        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split("/")[-1].split("-")[-1]
                    x_reshaped_v = np.reshape(mnist.validation.images, [-1, 
                                                mnist_inference.IMAGE_SIZE,
                                                mnist_inference.IMAGE_SIZE, 
                                                mnist_inference.IMAGE_CHANNELS])
                    x_reshaped_t = np.reshape(mnist.test.images, [-1, 
                                                mnist_inference.IMAGE_SIZE,
                                                mnist_inference.IMAGE_SIZE, 
                                                mnist_inference.IMAGE_CHANNELS])
                    vali_acc = sess.run(accuracy, feed_dict={x: x_reshaped_v, y_: mnist.validation.labels})
                    test_acc = sess.run(accuracy, feed_dict={x: x_reshaped_t, y_: mnist.test.labels})
                    print("After %s steps, the accuracy on validation: %g, the accuracy on test: %g." % (global_step, vali_acc, test_acc))
                else:
                    print("checkpoint model not found!")
                    return
            time.sleep(EVAL_INTERVAL_SECS)

def main(argv=None):
    mnist = input_data.read_data_sets("../../part 5/mnist/", one_hot=True)
    evaluate(mnist)
    
if __name__ == "__main__":
    tf.app.run()