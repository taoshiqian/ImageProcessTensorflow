import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

# 生成整数型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# 生成字符型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

mnist = input_data.read_data_sets(
    "E:\\pycode\\ImageProcessTensorflow\\LeNet5_mnist\\mnist_data",dtype=tf.uint8,one_hot=True
)

images = mnist.train.images
# labels，做属性
labels = mnist.train.labels
# 图像分辨率，做属性
pixels = images.shape[1]
num_examples = mnist.train.num_examples

#TFRecord文件路径
filename = "output.tfrecords"
with tf.python_io.TFRecordWriter(filename) as writer:
    for index in range(num_examples):
        # 将图像矩阵转成一个字符串
        image_raw = images[index].tostring()
        # 将一个样例转化成Example Protocol Buffer,并将信息写入这个数据结构
        example = tf.train.Example(
            features = tf.train.Features(feature={
                'pixels': _int64_feature(pixels),
                'labels': _int64_feature(np.argmax(labels[index])),
                'image_raw': _bytes_feature(image_raw)
            })
        )
        # 将一个Example写入TFRecord文件
        writer.write(example.SerializeToString())