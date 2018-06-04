import tensorflow as tf

# 使用tensorflow原始API实现卷积层
with tf.variable_scope('tf-API'):
    weights = tf.get_variable("weights")
    biases = tf.get_variable("biases")
    conv = tf.nn.conv2d()
    relu = tf.nn.relu(tf.nn.bias_add(conv, biases))

# Slim
import tensorflow.contrib.slim as slim

# 输入节点矩阵，过滤器的深度，过滤器的尺寸
net = slim.conv2d(input, 32, [3, 3])
