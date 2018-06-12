import tensorflow as tf

# 通过文件列表创建输入文件队列
files = tf.train.match_filenames_once('data.tfrecords*')
filename_queue = tf.train.string_input_producer(files, shuffle=False)

# 解析TFRecord
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
    serialized_example, features={
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64),
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'channels': tf.FixedLenFeature([], tf.int64)
    }
)
image, label = features['image'], features['label']
height, width = features['height'], features['width']
channels = features['channels']

# 从原始图像到像素矩阵
decoded_image = tf.decode_raw(image, tf.uint8)
decoded_image.set_shape([height, width, channels])

# 神经网络输入层图像大小
image_size = 299

# 图像预处理.翻转，裁剪等等
distorted_image = preprocess_for_train(decoded_image, image_size, image_size, None)

# 将数据整理成batch
min_after_dequeue = 10000
batch_size = 100
capacity = min_after_dequeue + 3 * batch_size
image_batch, label_batch = tf.train.shuffle_batch(
    [distorted_image, label], batch_size=batch_size,
    capacity=capacity, min_after_dequeue=min_after_dequeue
)

# 定义神经网络
learning_rate = 0.01
logit = infrence(image_batch)
loss = calc_loss(logit, label_batch)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# 运行神经网络
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer(), tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # 训练
    TRAINING_ROUNDS = 5000
    for i in range(TRAINING_ROUNDS):
        sess.run(train_step)

    # 停止所有线程
    coord.request_stop()
    coord.join(threads)


