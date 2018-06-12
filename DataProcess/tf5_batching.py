import tensorflow as tf


# 2. 读取文件。
def readData():
    # 正则获取文件列表
    files = tf.train.match_filenames_once("data.tfrecords-*")
    # 输入队列
    filename_queue = tf.train.string_input_producer(files, shuffle=False)

    # 如图7.1节中所示，读取并解析一个样本
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'i': tf.FixedLenFeature([], tf.int64),
            'j': tf.FixedLenFeature([], tf.int64),
        })

    with tf.Session() as sess:
        tf.local_variables_initializer().run()
        print(sess.run(files))

        # 声明tf.train.Coordinator类来协同不同线程，并启动线程
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # 多次执行获取数据的操作
        for i in range(6):
            print(sess.run([features['i'], features['j']]))
        coord.request_stop()
        coord.join()
    return features


# 3.组合训练数据
def batching(features):
    # 特征与标签
    example, label = features['i'], features['j']

    batch_size = 3

    #队列容量
    capacity = 1000 + 3 * batch_size

    example_batch, label_batch = tf.train.batch([example, label], batch_size=batch_size, capacity=capacity)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(5):
            cur_example_batch, cur_label_batch = sess.run([example_batch, label_batch])
            print(cur_example_batch, cur_label_batch)
        coord.request_stop()
        coord.join(threads)

# 3.组合训练数据
def batching2(features):
    # 特征与标签
    example, label = features['i'], features['j']

    batch_size = 3

    #队列容量
    capacity = 1000 + 3 * batch_size

    example_batch, label_batch = tf.train.shuffle_batch(
        [example, label], batch_size=batch_size, capacity=capacity,min_after_dequeue=30)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(5):
            cur_example_batch, cur_label_batch = sess.run([example_batch, label_batch])
            print(cur_example_batch, cur_label_batch)
        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    # produce()
    features = readData()
    batching2(features)
