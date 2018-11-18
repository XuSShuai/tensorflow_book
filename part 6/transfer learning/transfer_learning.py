import tensorflow as tf
import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inception_v3
import numpy as np
import tensorflow.contrib.slim as slim

INPUT_DATA = "../processed_flower_data/processed_flower_data.npy"
CKPT_FILE = "../../inception_v3/inception_v3.ckpt"
MODEL_SAVE = "./model_save/transfer_model"

TRAINING_SCOPES = "InceptionV3/Logits,InceptionV3/AuxLogits"
N_CLASSES = 5
LEARNING_RATE = 0.0001
EPOCH = 300
BATCH_SIZE = 32

def get_tuned_variables():
    scopes = [scope.strip() for scope in TRAINING_SCOPES.split(",")]
    variables_to_restore = []
    for var in slim.get_model_variables():
        if not var.op.name.startswith(scopes[0]) and not var.op.name.startswith(scopes[1]):
            variables_to_restore.append(var)
    return variables_to_restore

def main(argv=None):
    processed_data = np.load(INPUT_DATA)
    train_image = processed_data[0]
    train_image_n = len(train_image)
    train_label = processed_data[1]
    validation_image = processed_data[2]
    validation_label = processed_data[3]
    test_image = processed_data[4]
    test_label = processed_data[5]
    
    images = tf.placeholder(tf.float32, [None, 299, 299, 3], name="input_images")
    labels = tf.placeholder(tf.int64, [None], name="labels")
    
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits, _ = inception_v3.inception_v3(images, num_classes=N_CLASSES)
        
    scopes = [scope.strip() for scope in TRAINING_SCOPES.split(",")]
    trainable_variables = []
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        trainable_variables.append(variables)
    
    tf.losses.softmax_cross_entropy(tf.one_hot(labels, N_CLASSES), logits)
    
    train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(tf.losses.get_total_loss())
    
    with tf.name_scope("evaluation"):
        evalution_op = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), labels), tf.float32))
    
    load_model_op = slim.assign_from_checkpoint_fn(CKPT_FILE, get_tuned_variables(), ignore_missing_vars = True)
    
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        
        print("load tuned variables from %s " % (CKPT_FILE))
        load_model_op(sess)
        
        for i in range(EPOCH):
            start = (i * BATCH_SIZE) % train_image_n
            end = min(start + BATCH_SIZE, train_image_n)
            sess.run(train_op, feed_dict={images: train_image[start: end], labels: train_label[start: end]})
            if i % 30 == 0:
                saver.save(sess, MODEL_SAVE, global_step = i)
                train_acc = sess.run(evalution_op, feed_dict = {images: train_image, labels: train_label})
                validation_acc = sess.run(evalution_op, feed_dict = {images: validation_image, labels: validation_label})
                print("After %d step, train acc: %f, validation acc: %f." % (i, train_acc, validation_acc))
        test_acc = sess.run(evalution_op, feed_dict = {images: test_image, labels: test_label})
        print("final test acc: %f" % (test_acc))

if __name__ == "__main__":
    tf.app.run()
