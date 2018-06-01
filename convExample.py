# coding:utf-8

import tensorflow as tf

# 过滤器的参数
filter_weight = tf.get_variable(
    'weights', [5, 5, 3, 16],  # 过滤器的尺寸，，当层的深度，过滤器的深度(也是下一层节点矩阵的深度)
    initializer=tf.truncated_normal_initializer(stddev=0.1)
)
# 偏置参数
biases = tf.get_variable(
    'biases', [16],
    initializer=tf.truncated_normal_initializer(0.1)
)

# 卷积层
conv = tf.nn.conv2d(
    input  # 当前层的节点矩阵。第一维对应一个输入batch，后三维对应一个节点矩阵(如一个图像的尺寸)
    , filter_weight  # 卷积层参数
    , strides=[1, 1, 1, 1]  # 不同维度上的不唱，一四只能是1，二三表示在矩阵长和宽上的步长
    , padding='SAME'  # SAME全0填充，VALID不添加
)

# 加上偏置项
bias = tf.nn.bias_add(conv, biases)

# 激活函数得到最终结果
actived_conv = tf.nn.relu(bias)

# max pooling
pool = tf.nn.max_pool(actived_conv,
                      ksize=[1,3,3,1],strides=[1,2,2,1]
                      ,padding='SAME')