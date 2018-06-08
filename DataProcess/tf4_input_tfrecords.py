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

if __name__ == '__main__':
    # produce()
    pass