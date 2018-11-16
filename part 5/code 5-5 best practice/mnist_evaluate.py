import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference
import mnist_train
import time

EVAL_INTERVAL_SECS = 10

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(dtype=tf.float32, shape=[None, mnist_inference.INPUT_NODE], name="x-input")
        y_ = tf.placeholder(dtype=tf.float32, shape=[None, mnist_inference.OUTPUT_NODE], name="y-input")

        y = mnist_inference.inference(x, None)

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
                    validation_acc = sess.run(accuracy, feed_dict={x: mnist.validation.images, y_: mnist.validation.labels})
                    print("After %s steps, the accuracy on validation set is %g." % (global_step, validation_acc))
                else:
                    print("checkpoint model not found!")
            time.sleep(EVAL_INTERVAL_SECS)

def main(argv=None):
    mnist = input_data.read_data_sets("../mnist/", one_hot=True)
    evaluate(mnist)
    
if __name__ == "__main__":
    tf.app.run()