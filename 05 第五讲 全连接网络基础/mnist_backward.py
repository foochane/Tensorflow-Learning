#coding:utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import os

BATCH_SIZE = 200  #每轮喂入神经网络的图片量
LEARNING_RATE_BASE = 0.1  #学习率
LEARNING_RATE_DECAY = 0.99  #衰减率
REGULARIZER = 0.0001   #正则化系数
STEPS = 50000   #训练总轮数
MOVING_AVERAGE_DECAY = 0.99  #滑动平均衰减率
MODEL_SAVE_PATH="./model/"  #模型的保存路径
MODEL_NAME="mnist_model"   #模型保存的文件名


def backward(mnist):

    #给x和y_占位
    x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
    y_ = tf.placeholder(tf.float32, [None, mnist_forward.OUTPUT_NODE])
    #调用前向传播的程序，计算输出y
    y = mnist_forward.forward(x, REGULARIZER)
    #轮数计数器赋初始值，设定为不可训练
    global_step = tf.Variable(0, trainable=False)

    #调用包含正则化的损失函数loss
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection('losses'))

    #定义指数衰减学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE, 
        LEARNING_RATE_DECAY,
        staircase=True)

    #定义训练过程
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    #定义滑动平均
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')

    #实例化saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        #初始化所有变量
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        for i in range(STEPS):
            #每次读入BATCH_SIZE组图片和标签，并喂入神经网络，执行训练过程
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})

            #每1000轮打印出当前的loss值
            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main():
    mnist = input_data.read_data_sets("./data/", one_hot=True)
    backward(mnist)

if __name__ == '__main__':
    main()


