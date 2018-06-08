import tensorflow as tf
import numpy as np
import threading
import time


# 每隔一秒打印自己的ID
def MyLoop(coord, worker_id):
    # 不需要停止
    while not coord.should_stop():
        # 随机停止所有线程
        if np.random.rand() < 0.1:
            print("Stoping from id: %d"%worker_id)
            # 通知其他线程停止。其他人的should_stop会变成True
            coord.request_stop()
        else:
            # 打印ID
            print("Working on id: %d"%worker_id)
        # 暂停1秒
        time.sleep(1)

if __name__ == '__main__':
    # Coordinator协调者。用来协同多个线程
    coord = tf.train.Coordinator()
    # 声明5个线程
    threads = [
        threading.Thread(target=MyLoop, args=(coord,i,)) for i in range(5)
    ]
    # 启动所有线程
    for t in threads:
        t.start()
    # 等待所有线程退出...join(<list of threads>):等待被指定的线程终止。
    coord.join(threads)