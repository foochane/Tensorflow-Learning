#coding:utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_lenet5_forward
import os
import numpy as np

BATCH_SIZE = 100 # 一个 batch 的数量
LEARNING_RATE_BASE =  0.005 
LEARNING_RATE_DECAY = 0.99 # 学习率的衰减率
REGULARIZER = 0.0001 # 正则化项的权重
STEPS = 50000 # 最大迭代次数
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均的衰减率
MODEL_SAVE_PATH="./model/" # 保存模型的路径
MODEL_NAME="mnist_model" # 模型命名

def backward(mnist):
    x = tf.placeholder(tf.float32,[
	BATCH_SIZE,
	mnist_lenet5_forward.IMAGE_SIZE,
	mnist_lenet5_forward.IMAGE_SIZE,
	mnist_lenet5_forward.NUM_CHANNELS]) 
    y_ = tf.placeholder(tf.float32, [None, mnist_lenet5_forward.OUTPUT_NODE])
    y = mnist_lenet5_forward.forward(x,True, REGULARIZER) 
    global_step = tf.Variable(0, trainable=False) 

    # 先是对网络最后一层的输出 y 做 softmax,通常是求取输出属于某一类的概率,其实就是一个num_classes 大小的向量,
    # 再将此向量和实际标签值做交叉熵,需要说明的是该函数返回的是一个向量
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce) # 再对得到的向量求均值就得到 loss
    loss = cem + tf.add_n(tf.get_collection('losses')) # 添加正则化中的 losses

    # 实现指数级的减小学习率,可以让模型在训练的前期快速接近较优解,又可以保证模型在训练后期不会有太大波动
    # 计算公式:decayed_learning_rate=learining_rate*decay_rate^(global_step/decay_steps)
    learning_rate = tf.train.exponential_decay( 
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE, 
		LEARNING_RATE_DECAY,
        staircase=True) 
    
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 就是相应变量的初始值,每次变量更新时,影子变量就会随之更新
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]): 
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver() 

    with tf.Session() as sess: 
        init_op = tf.global_variables_initializer() 
        sess.run(init_op) 

        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH) 
        if ckpt and ckpt.model_checkpoint_path:
        	saver.restore(sess, ckpt.model_checkpoint_path) 

        for i in range(STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE) 
            reshaped_xs = np.reshape(xs,(  
		    BATCH_SIZE,
        	mnist_lenet5_forward.IMAGE_SIZE,
        	mnist_lenet5_forward.IMAGE_SIZE,
        	mnist_lenet5_forward.NUM_CHANNELS))
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys}) 
            if i % 100 == 0: 
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

def main():
    mnist = input_data.read_data_sets("./data/", one_hot=True) 
    backward(mnist)

if __name__ == '__main__':
    main()


