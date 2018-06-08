import tensorflow as tf

queue = tf.FIFOQueue(100, "float")

enqueue_op = queue.enqueue([tf.random_normal([1])])

# 创建5个线程，每个线程都是对queue执行enqueue_op操作
qr = tf.train.QueueRunner(queue, [enqueue_op] * 5)

# 将 QueueRunner 加入计算图
tf.train.add_queue_runner(qr)

# 定义出队操作
out_tensor = queue.dequeue()

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    # 启动计算图。即启动所有线程
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # 获取队列中的取值
    for _ in range(3):
        print(sess.run(out_tensor)[0])

    # 停止所有线程
    coord.request_stop()
    coord.join(threads)
