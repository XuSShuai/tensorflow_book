import tensorflow as tf

IMAGE_SIZE = 28
IMAGE_CHANNELS = 1
CONV1_SIZE = 5
CONV1_NUM = 32
CONV2_SIZE = 5
CONV2_NUM = 64
FC_SIZE = 512
LABELS = 10


def get_weight(shape, regularizer):
    weight = tf.get_variable(name="weight", shape=shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer:
        tf.add_to_collection("losses", regularizer(weight))
    return weight

def get_bias(shape):
    bias = tf.get_variable(name="bias", shape=shape, initializer=tf.zeros_initializer())
    return bias

def inference(input_tensor, train, regularizer):
    with tf.variable_scope("layer1-conv1"):
        weight = get_weight([CONV1_SIZE, CONV1_SIZE, IMAGE_CHANNELS, CONV1_NUM], regularizer)
        bias = get_bias([CONV1_NUM])
        conv1 = tf.nn.conv2d(input_tensor, weight, strides=[1, 1, 1, 1], padding="SAME")
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, bias))
    
    with tf.variable_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        
    with tf.variable_scope("layer3-conv2"):
        weight = get_weight([CONV2_SIZE, CONV2_SIZE, CONV1_NUM, CONV2_NUM], regularizer)
        bias = get_bias([CONV2_NUM])
        conv2 = tf.nn.conv2d(pool1, weight, strides=[1, 1, 1, 1], padding="SAME")
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, bias))
        
    with tf.variable_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    
    shape = pool2.get_shape().as_list()
    nodes = shape[1] * shape[2] * shape[3]
    
    reshaped = tf.reshape(pool2, [-1, nodes])
    
    with tf.variable_scope("layer5-fc1"):
        weight = get_weight([nodes, FC_SIZE], regularizer)
        bias = get_bias([FC_SIZE])
        fc1 = tf.nn.relu(tf.matmul(reshaped, weight) + bias)
        if train:
            fc1 = tf.nn.dropout(fc1, keep_prob=0.5)
    
    with tf.variable_scope("layer6-fc2"):
        weight = get_weight([FC_SIZE, LABELS], regularizer)
        bias = get_bias([LABELS])
        fc2 = tf.matmul(fc1, weight) + bias
    
    return fc2