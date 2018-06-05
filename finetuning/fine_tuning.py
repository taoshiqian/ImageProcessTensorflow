# -*- coding: utf-8 -*-

import glob  # 查找文件
import os.path
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import tensorflow.contrib.slim as slim

# 加载通过tensorflow-slim定义好的inception-v3模型
import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inception_v3

# 处理之后的数据
INPUT_DATA = 'flower_processed_data.npy'
# 保存训练好的模型；将新数据训练得到的完整模型保存下来
TRAIN_FILE = 'save_model'
# 谷歌训好的inception-V3模型
CKPT_FILE = 'inception_v3.ckpt'

# 训练的超参数
LEARNING_RATE = 0.0001
STEPS = 300
BATCH = 32
N_CLASSES = 5

# 最后的全连接层，需训练，不从谷歌模型中加载
CHECKPOINT_EXCLUDE_SCOPES = 'InceptionV3/Logits,Inception/AuxLogits'
# 需要训练的网络参数名称（这里是参数的后缀）
# TRAINABLE_SCOPES = 'InceptionV3/Logits,InceptionV3/AuxLogits'
TRAINABLE_SCOPES = 'InceptionV3/Logits,InceptionV3/AuxLogit'


# 获取所有需要从谷歌模型中加载的参数
def get_tuned_variables():
    exclusions = [scope.strip() for scope in CHECKPOINT_EXCLUDE_SCOPES.split(',')]
    variables_to_train = []

    # 枚举inception-v3模型中所有的参数，然后判断是否需要从加载列表中移除
    for var in slim.get_model_variables():
        excluded = False  # 不排除
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):  # 前缀
                excluded = True
                break
        if not excluded:  # 不排除
            variables_to_train.append(var)
    return variables_to_train


# 获取所有需要训练的变量列表
def get_trainable_variables():
    scopes = [scope.strip() for scope in TRAINABLE_SCOPES.split(',')]
    variables_to_train = []
    # 枚举所有需要训练的参数前缀，并通过这些前缀找到所有的参数
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    return variables_to_train


def main():
    # 加载预处理好的数据
    processed_data = np.load(INPUT_DATA)
    training_images = processed_data[0]
    n_training_example = len(training_images)
    training_labels = processed_data[1]
    validation_images = processed_data[2]
    validation_labels = processed_data[3]
    testing_images = processed_data[4]
    testing_labels = processed_data[5]
    print("%d training examples, %d validation examples and %d testing examples." % (
        n_training_example, len(validation_labels), len(testing_labels)))

    # 定义inception-v3的输入，images为输入图片，labels为每一张图片对应的标签
    images = tf.placeholder(tf.float32, [None, 399, 399, 3], name='input_images')
    labels = tf.placeholder(tf.int64, [None], name='labels')

    # 定义inception-v3网络结构
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits, _ = inception_v3.inception_v3(images, num_classes=N_CLASSES)

    # 获取需要训练的变量
    training_variables = get_trainable_variables()
    # 定义交叉熵损失。注意在模型定义的时候已经将正则化损失加入损失集合了
    tf.losses.softmax_cross_entropy(tf.one_hot(labels, N_CLASSES), logits, weights=1.0)
    # 定义训练过程.
    train_step = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(tf.losses.get_total_loss())

    # 计算正确率
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 定义加载模型的函数
    load_fn = slim.assign_from_checkpoint_fn(
        CKPT_FILE,
        get_tuned_variables(),
        ignore_missing_vars=True
    )

    # 保存新模型
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # 初始化没有加载进来的变量。一定要在模型加载之前初始化，否则初始化会影响加载进来的模型。
        init = tf.global_variables_initializer()
        sess.run(init)

        # 加载谷歌训练好的模型
        print('Loading tuned variables from %s' % CKPT_FILE)
        load_fn(sess)

        start = 0
        end = BATCH
        for i in range(STEPS):
            # 运行训练过程，这里不会更新全部参数，只会更新制定的部分参数
            _, loss = sess.run(train_step, feed_dict={
                images: training_images[start:end],
                labels: training_labels[start:end]
            })
            # 输出日志
            if i % 30 == 0 or i + 1 == STEPS:
                saver.save(sess, TRAIN_FILE, global_step=i)

                validation_accuracy = sess.run(evaluation_step, feed_dict={
                    images: validation_images, labels: validation_labels})
                print('Step %d: Training loss is %.1f Validation accuracy = %.1f%%' % (
                    i, loss, validation_accuracy * 100.0))

            start = end
            if start == n_training_example:
                start = 0

            end = start + BATCH
            if end > n_training_example:
                end = n_training_example

    # 在最后的测试数据上测试正确率。
    test_accuracy = sess.run(evaluation_step, feed_dict={
        images: testing_images, labels: testing_labels})
    print('Final test accuracy = %.1f%%' % (test_accuracy * 100))


if __name__ == '__main__':
    tf.app.run()
