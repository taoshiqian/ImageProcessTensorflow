# -*-  coding : utf-8  -*-
# 变换以及展示
import matplotlib.pyplot as plt
import tensorflow as tf

image_raw_data = tf.gfile.FastGFile("cat.jpg", "rb").read()
image_data = tf.image.decode_jpeg(image_raw_data)
image_float = tf.image.convert_image_dtype(image_data, tf.float32)


with tf.Session() as sess:
    plt.imshow(image_float.eval())
    plt.show()


with tf.Session() as sess:
    adjusted = tf.image.per_image_standardization(image_float)
    adjusted = tf.clip_by_value(adjusted, 0.0, 1.0)
    plt.imshow(adjusted.eval())
    plt.show()




with tf.Session() as sess:
    adjusted = tf.image.adjust_saturation(image_float,-2)
    adjusted = tf.clip_by_value(adjusted, 0.0, 1.0)
    plt.imshow(adjusted.eval())
    plt.show()


with tf.Session() as sess:
    adjusted = tf.image.adjust_saturation(image_float,2)
    adjusted = tf.clip_by_value(adjusted, 0.0, 1.0)
    plt.imshow(adjusted.eval())
    plt.show()



with tf.Session() as sess:
    adjusted = tf.image.adjust_hue(image_float,0.1)
    adjusted = tf.clip_by_value(adjusted, 0.0, 1.0)
    plt.imshow(adjusted.eval())
    plt.show()

with tf.Session() as sess:
    adjusted = tf.image.adjust_hue(image_float,0.5)
    adjusted = tf.clip_by_value(adjusted, 0.0, 1.0)
    plt.imshow(adjusted.eval())
    plt.show()

with tf.Session() as sess:
    adjusted = tf.image.adjust_hue(image_float,0.9)
    adjusted = tf.clip_by_value(adjusted, 0.0, 1.0)
    plt.imshow(adjusted.eval())
    plt.show()





with tf.Session() as sess:
    adjusted = tf.image.adjust_contrast(image_float,0.5)
    adjusted = tf.clip_by_value(adjusted, 0.0, 1.0)
    plt.imshow(adjusted.eval())
    plt.show()

with tf.Session() as sess:
    adjusted = tf.image.adjust_contrast(image_float,2)
    adjusted = tf.clip_by_value(adjusted, 0.0, 1.0)
    plt.imshow(adjusted.eval())
    plt.show()






with tf.Session() as sess:
    adjusted = tf.image.adjust_brightness(image_float,-0.5)
    adjusted = tf.clip_by_value(adjusted, 0.0, 1.0)
    plt.imshow(adjusted.eval())
    plt.show()




with tf.Session() as sess:
    flipped = tf.image.flip_up_down(image_float)
    plt.imshow(flipped.eval())
    plt.show()

with tf.Session() as sess:
    flipped = tf.image.flip_left_right(image_float)
    plt.imshow(flipped.eval())
    plt.show()

with tf.Session() as sess:
    flipped = tf.image.transpose_image(image_float)
    plt.imshow(flipped.eval())
    plt.show()











with tf.Session() as sess:
    plt.imshow(image_float.eval())
    plt.show()

with tf.Session() as sess:
    resized = tf.image.resize_images(image_float, [300, 300], method=0)
    plt.imshow(resized.eval())
    plt.show()

with tf.Session() as sess:
    croped = tf.image.resize_image_with_crop_or_pad(image_float, 200, 200)
    plt.imshow(croped.eval())
    plt.show()
    paded = tf.image.resize_image_with_crop_or_pad(image_float, 800, 800)
    plt.imshow(paded.eval())
    plt.show()

with tf.Session() as sess:
    central_cropped = tf.image.central_crop(image_float,0.5)
    plt.imshow(central_cropped.eval())
    plt.show()

