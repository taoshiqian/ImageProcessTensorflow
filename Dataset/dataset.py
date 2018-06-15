# -*- coding:utf-8 -*-
# Dataset的使用

import tensorflow as tf


# 数组到Dataset
def test1_arr():
    # 从一个数组创建数据集
    input_data = [1, 2, 3, 5, 8]
    # 类似于入队列
    dataset = tf.data.Dataset.from_tensor_slices(input_data)

    # 定义迭代器。
    iterator = dataset.make_one_shot_iterator()

    # get_next() 返回代表一个输入数据的张量。类似于出队列
    x = iterator.get_next()
    y = x * x

    with tf.Session() as sess:
        for i in range(len(input_data)):
            print(sess.run(y))


# 文本到Dataset
def test2_txt():
    # 创建文本文件作为本例的输入
    with open('./test1.txt', 'w') as file:
        file.write("File1 line1 \n")
        file.write("File1 lene2 \n")
    with open('./test2.txt', 'w') as file:
        file.write("File2 line1 \n")
        file.write("File2 lene2 \n")

    # 以文本文件创建数据集
    input_files = ["./test1.txt", "./test2.txt"]
    dataset = tf.data.TextLineDataset(input_files)

    # 定义迭代器用于遍历数据集
    iterator = dataset.make_one_shot_iterator()

    # 这里get_next()返回一个字符串类型的张量，代表文件中的一行。
    x = iterator.get_next()

    with tf.Session() as sess:
        for i in range(4):
            print(sess.run(x))


# 解析一个TFRecord的方法。# 解析一个T
def parser(record):
    features = tf.parse_single_example(
        record,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'pixels': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64)
        })
    decoded_images = tf.decode_raw(features['image_raw'], tf.uint8)
    retyped_images = tf.cast(decoded_images, tf.float32)
    images = tf.reshape(retyped_images, [784])
    labels = tf.cast(features['label'], tf.int32)
    # pixels = tf.cast(features['pixels'],tf.int32)
    return images, labels


# tfrecord到dataset
def test3_tfrecord():
    # 从TFRecord文件创建数据集
    input_files = ["output.tfrecords"]
    dataset = tf.data.TFRecordDataset(input_files)
    # map()函数表示对数据集中的每一条数据进行调用解析方法。
    dataset = dataset.map(parser)
    # 定义遍历数据集的迭代器
    iterator = dataset.make_one_shot_iterator()
    # 读取数据，可用于进一步计算
    image, label = iterator.get_next()

    with tf.Session() as sess:
        for i in range(10):
            x, y = sess.run([image, label])
            print(y)


# 使用initializable_iterator来动态初始化数据集。可以用placeholder
def test4_initializable_iterator():
    input_files = tf.placeholder(tf.string)
    dataset = tf.data.TFRecordDataset(input_files)
    dataset = dataset.map(parser)

    # 定义遍历dataset的迭代器
    iterator = dataset.make_initializable()
    image, label = iterator.get_next()

    with tf.Session() as sess:
        # 初始化iterator，并给出input_file的值
        sess.run(iterator.initializer,
                 feed_dict={input_files:['output.tf.record']})
        # 遍历
        while True:
            try:
                x, y = sess.run([image, label])
            except:
                break


if __name__ == '__main__':
    test3_tfrecord()
