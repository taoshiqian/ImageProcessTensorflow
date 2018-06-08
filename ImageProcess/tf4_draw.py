# -*-  coding : utf-8  -*-
# 画标注
import matplotlib.pyplot as plt
import tensorflow as tf

image_raw_data = tf.gfile.FastGFile("cat.jpg", "rb").read()
image_data = tf.image.decode_jpeg(image_raw_data)
image_float = tf.image.convert_image_dtype(image_data, tf.float32)

with tf.Session() as sess:
    plt.imshow(image_float.eval())
    plt.show()

# 画标注框
with tf.Session() as sess:
    batched = tf.expand_dims(image_float, 0)
    boxes = tf.constant([[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]])
    result = tf.image.draw_bounding_boxes(batched, boxes=boxes)
    plt.imshow(result[0].eval())
    plt.show()

# 截取部分
# with tf.Session() as sess:
#     boxes = tf.constant([[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]])
#     begin,size,bbox_for_draw = tf.image.sample_distorted_bounding_box(
#         tf.shape(image_float), bounding_boxes=boxes,
#         min_object_covered=0.4
#     )
#     batched = tf.expand_dims(image_float,0)
#     image_with_box = tf.image.draw_bounding_boxes(image_float, boxes)
#     plt.imshow(image_with_box[0].eval())
#     plt.show()
#     distorted_image = tf.slice(image_float, begin, size)
#     plt.imshow(distorted_image[0].eval())
#     plt.show()