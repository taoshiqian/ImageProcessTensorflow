# coding:utf-8

import tensorflow as tf

# 配置神经网络参数
INPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

# 第一层卷积层的深度与尺度
CONV1_DEEP = 32
CONV1_SIZE = 5
# 第二层卷积层的深度与尺寸
CONV2_DEEP = 64
CONV2_SIZE = 5
# 全连接层的节点个数
FC_SIZE = 512


# 定义卷积神经网络的前向传播（即网络结构）。
# train用于区分是训练过程还是测试过程
def inference(input_tensor, train, regularizer):
    # 1.卷积层1 + relu
    # 28*28*1 -> 28*28*32
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable(
            "weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        conv1_biases = tf.get_variable(
            'bias', [CONV1_DEEP],
            initializer=tf.constant_initializer(0.0)
        )
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    # 2.池化层1
    # 28*28*32 -> 14*14*32
    with tf.name_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(
            relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'
        )

    # 3.卷积层2 + relu
    # 14*14*32 ->14*14*64
    with tf.variable_scope('layer3-conv2'):
        conv2_weights = tf.get_variable(
            "weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        conv2_biases = tf.get_variable(
            "bias", [CONV2_DEEP],
            initializer=tf.constant_initializer(0.0)
        )
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, bias=conv2_biases))

    # 4.池化层2
    # 14*14*64 -> 7*7*64
    with tf.name_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(
            relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'
        )
        # 把7*7*64拉成一个向量。0：batch数据个数，1：长度，2：宽度，3：深度
        pool_shape = pool2.get_shape().as_list()
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    # 5.全连接1 + relu （dropout一般只在全连接层使用，而不在卷积层或者池化层使用；只有全连接层的权重需要加入正则化项）
    # 3136 -> 512
    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable(
            "weight", [nodes, FC_SIZE],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        # 只有全连接层的权重需要加入正则化项
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable(
            'bias', [FC_SIZE],
            initializer=tf.constant_initializer(0.1)
        )
        fc1 = tf.nn.relu(tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases))
        if train: fc1 = tf.nn.dropout(fc1, 0.5)  # dropout一般只在全连接层使用，而不在卷积层或者池化层使用

    # 6.全连接2
    # 512 -> 10
    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable(
            "weight", [FC_SIZE, NUM_LABELS],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable(
            "bias", [NUM_LABELS],
            initializer=tf.constant_initializer(0.1)
        )
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases

    return logit
