import tensorflow as tf
import parameters

def conv2d(inputs, weights, biases, strides=1):
    inputs = tf.nn.conv2d(inputs, weights, [1, 1, 1,1],padding = 'SAME')
    inputs = tf.nn.bias_add(inputs,biases)
    return tf.nn.relu(inputs)

def maxpool2d(inputs,k=2):
    return tf.nn.max_pool(inputs, [1,k,k,1],[1,k,k,1],padding='SAME')

def convnet(inputs, weights, biases, keep_prob = parameters.dropout):
    # Layer 1
    conv1 = conv2d(inputs,weights['wc1'],biases['bc1'])
    conv1 = maxpool2d(conv1)

    # Layer 2
    conv2 = conv2d(conv1,weights['wc2'],biases['bc2'])
    conv2 = maxpool2d(conv2)

    # Fully Connected Layer
    fc1 = tf.reshape(conv2, [-1,weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1,weights['wd1']),biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    # Apply Dropout
    fc1 = tf.nn.dropout(fc1,keep_prob)

    # Output
    return tf.add(tf.matmul(fc1,weights['out']),biases['out'],name='logits')
