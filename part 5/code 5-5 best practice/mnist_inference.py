import tensorflow as tf

INPUT_NODE = 784
OUTPUT_NODE = 10
HIDDEN_NODE = 500

def get_weight(shape, regularizer):
    weight = tf.get_variable(name="weight", shape=shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer:
        tf.add_to_collection("losses", regularizer(weight))
    return weight

def get_bias(shape):
    return tf.get_variable(name="bias", shape=shape, initializer=tf.zeros_initializer())
    
def inference(input_tensor, regularizer):
    with tf.variable_scope("layer1"):
        weight = get_weight([INPUT_NODE, HIDDEN_NODE], regularizer)
        bias = get_bias([HIDDEN_NODE])
        a1 = tf.nn.relu(tf.matmul(input_tensor, weight) + bias)
    with tf.variable_scope("layer2"):
        weight = get_weight([HIDDEN_NODE, OUTPUT_NODE], regularizer)
        bias = get_bias([OUTPUT_NODE])
        y = tf.matmul(a1, weight) + bias
    return y