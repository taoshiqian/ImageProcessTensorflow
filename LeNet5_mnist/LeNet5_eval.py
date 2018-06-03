# coding:
import numpy as np
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from LeNet5_mnist import LeNet5_inference
from LeNet5_mnist import LeNet5_train

# 每10秒加载一次模型，并在最新模型上测试准确率
EVAL_INTERVAL_SECS = 10


def evaluate(mnist):
    with tf.Graph().as_default() as g:
        # 定义输入输出的格式
        x = tf.placeholder(
            tf.float32,
            [None, LeNet5_inference.IMAGE_SIZE, LeNet5_inference.IMAGE_SIZE, LeNet5_inference.NUM_CHANNELS],
            name='x-input'
        )
        y_ = tf.placeholder(tf.float32, [None, LeNet5_inference.OUTPUT_NODE], name='y-input')
        #reshaped_x = np.reshape(mnist.validation.images,(None, LeNet5_inference.IMAGE_SIZE, LeNet5_inference.IMAGE_SIZE, LeNet5_inference.NUM_CHANNELS))
        validate_feed = {
            x: mnist.validation.images,
            y_: mnist.validation.labels
        }

        # 直接调用inference的网络，不需要正则化项
        y = LeNet5_inference.inference(x, train=False,regularizer=None)

        # 准确率。
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 通过变量重命名的方式来加载模型。
        variable_average = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_average.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        # 每隔EVAL_INTERVAL_SECS秒调用一次检测
        while True:
            with tf.Session() as sess:
                # get_checkpoint_state通过checkpoint文件自动找到目录中最新模型的文件名
                ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    # 加载模型
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    # 通过文件名得到模型保存时迭代的次数
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print(
                        "After %s training step(s), validation accuracy = %g %%" % (global_step, accuracy_score * 100))
                else:
                    print('No checkpoint file ')
                    return
            time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None):
    mnist = input_data.read_data_sets('mnist_data', one_hot=True)
    evaluate(mnist)


if __name__ == '__main__':
    tf.app.run()
