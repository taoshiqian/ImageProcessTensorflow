# -*-  coding : utf-8  -*-

import matplotlib.pyplot as plt
import tensorflow as tf

image_raw_data = tf.gfile.FastGFile("cat.jpg", "rb").read()
image_data = tf.image.decode_jpeg(image_raw_data)
image_float = tf.image.convert_image_dtype(image_data, tf.float32)


with tf.Session() as sess:
    plt.imshow(image_float.eval())
    plt.show()


with tf.Session() as sess:
    batched = tf.expand_dims(image_float,0)
    boxes = tf.constant([[[0.1,0.2,0.3,0.4],[0.5,0.6,0.7,0.8]]])
    result = tf.image.draw_bounding_boxes(batched,boxes=boxes)
    plt.imshow(result[0].eval())
    plt.show()