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

#
with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='VALID'):
    # 上一层
    net = []

    with tf.variable_scope('Mixed_7c'):
        with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(net, 320, [1, 1], scope='Conv2d_0a_1x1')

        with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')
            branch_1 = tf.concat(3, [  # 根据哪一维度来拼接，3表示根据深度来拼接
                slim.conv2d(branch_1, 38, [1, 3], scope='Conv2d_0b_1x3'),
                slim.conv2d(branch_1, 384, [3, 1], scope='Conv2d_0c_3x1')
            ])

        with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net, 448, [1, 1], scope='Conv2d_0a_1x1')
            branch_2 = slim.conv2d(net, 384, [3, 3], scope='Conv2d_0b_3x3')
            branch_2 = tf.concat(3, [
                slim.conv2d(branch_2, 384, [1, 3], scope='Conv2d_0c_1x3'),
                slim.conv2d(branch_2, 384, [3, 1], scope='Conv2d_0d_3x1')
            ])

        with tf.variable_scope('Branch_3'):
            branch_3 = slim.avg_pool2d(net,[3,3],scope='AvgPool_0a_3x3')
            branch_3 = slim.avg_pool2d(branch_3,192,[1,1],scope='Conv2d_0b_1x1')

        net = tf.concat(3,[
            branch_0,
            branch_1,
            branch_2,
            branch_3
        ])
