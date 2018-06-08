import tensorflow as tf

# 先进先出队列，容量2，类型int
q = tf.FIFOQueue(2, "int32")

# 初始化操作
init = q.enqueue_many(([1, 10],))

# 使用Dequeue将队列的第一个元素出队，并存储在变量x中
x = q.dequeue()

y = x + 1

# 加1后的值在入队列。到此完成计算图
q_inc = q.enqueue([y])

with tf.Session() as sess:
    # 运行初始化操作
    init.run()
    for _ in range(5):
        # 运行计算图
        v, _ = sess.run([x, q_inc])
        # 输出队头
        print(v)
