import tensorflow as tf


# 创建TFRecord文件的帮助函数
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 1. 生成文件存储样例数据。
def produce():
    # 总计写入多少个文件
    num_shards = 2
    # instances_per_shard定义了每个文件中有多少个数据
    instances_per_shard = 2

    for i in range(num_shards):
        filename = ('data.tfrecords-%.5d-of-%.5d' % (i, num_shards))
        writer = tf.python_io.TFRecordWriter(filename)
        # 将数据封装成Example结构并写入TFRecord文件
        for j in range(instances_per_shard):
            # Example 结构仅包含当前样例属于第几个文件以及是当前文件的第几个样本
            example = tf.train.Example(features=tf.train.Features(feature={
                'i': _int64_feature(i),
                'j': _int64_feature(j)}))
            writer.write(example.SerializeToString())
        writer.close()


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


if __name__ == '__main__':
    # produce()
    readData()
