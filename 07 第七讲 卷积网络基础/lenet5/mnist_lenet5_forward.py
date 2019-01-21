#coding:utf-8
import tensorflow as tf

# 定义神经网络可以接收的图片的尺寸和通道数
IMAGE_SIZE = 28
NUM_CHANNELS = 1

# 定义第一层卷积核的大小和个数
CONV1_SIZE = 5
CONV1_KERNEL_NUM = 32

# 定义第二层卷积核的大小和个数
CONV2_SIZE = 5
CONV2_KERNEL_NUM = 64

# 定义第三层全连接层的神经元个数
FC_SIZE = 512
OUTPUT_NODE = 10

# 定义初始化网络权重函数
def get_weight(shape, regularizer):
	w = tf.Variable(tf.truncated_normal(shape,stddev=0.1))
	if regularizer != None: tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w)) 
	return w

# 定义初始化偏置项函数
def get_bias(shape): 
	b = tf.Variable(tf.zeros(shape))  
	return b

def conv2d(x,w):  
    # strides 表示卷积核在不同维度上的移动步长为 1,第一维和第四维一定是 1,这是因为卷积层的步长只对矩阵的长和宽有效;
    # padding='SAME'表示使用全 0 填充,而'VALID'表示不填充
	return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):  
    # ksize 表示池化过滤器的边长为 2,strides 表示过滤器移动步长是 2,'SAME'提供使用全 0 填充
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') 

def forward(x, train, regularizer):
    # 实现第一层卷积层的前向传播过程
    conv1_w = get_weight([CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_KERNEL_NUM], regularizer) 
    conv1_b = get_bias([CONV1_KERNEL_NUM]) 
    conv1 = conv2d(x, conv1_w) 
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b)) 
    pool1 = max_pool_2x2(relu1) 

    # 实现第二层卷积层的前向传播过程,并初始化卷积层的对应变量
    conv2_w = get_weight([CONV2_SIZE, CONV2_SIZE, CONV1_KERNEL_NUM, CONV2_KERNEL_NUM],regularizer) 
    conv2_b = get_bias([CONV2_KERNEL_NUM])
    conv2 = conv2d(pool1, conv2_w) 
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))
    pool2 = max_pool_2x2(relu2)

    # 将上一池化层的输出 pool2(矩阵)转化为下一层全连接层的输入格式(向量)
    pool_shape = pool2.get_shape().as_list() 
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3] 
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes]) 

    # 实现第三层全连接层的前向传播过程
    fc1_w = get_weight([nodes, FC_SIZE], regularizer) # 初始化全连接层的权重,并加入正则化
    fc1_b = get_bias([FC_SIZE]) # 初始化全连接层的偏置项
    fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_w) + fc1_b) 
    if train: fc1 = tf.nn.dropout(fc1, 0.5)

    # 实现第四层全连接层的前向传播过程,并初始化全连接层对应的变量
    fc2_w = get_weight([FC_SIZE, OUTPUT_NODE], regularizer)
    fc2_b = get_bias([OUTPUT_NODE])
    y = tf.matmul(fc1, fc2_w) + fc2_b
    return y 
