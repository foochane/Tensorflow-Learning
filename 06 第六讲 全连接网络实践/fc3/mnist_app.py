#coding:utf-8

import tensorflow as tf
import numpy as np
from PIL import Image
import mnist_backward
import mnist_forward

# 将符合神经网络输入要求的图片喂给复现的神经网络模型,输出预测值
def restore_model(testPicArr):
	#创建一个默认图
	with tf.Graph().as_default() as tg:
		x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
		y = mnist_forward.forward(x, None)
		preValue = tf.argmax(y, 1)

		# 实现滑动平均模型，参数MOVING_AVERAGE_DECAY 用于控制模型更新的速度。
		# 训练过程中会对每一个变量维护一个影子变量，影子变量的初始值就是相应变量的初始值。
		# 每次更新时影子变量会随之更新
		variable_averages = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
 		variables_to_restore = variable_averages.variables_to_restore()
 		saver = tf.train.Saver(variables_to_restore)

		with tf.Session() as sess:
			# 通过checkpoint文件定位到保存的模型
			ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess, ckpt.model_checkpoint_path)
		
				preValue = sess.run(preValue, feed_dict={x:testPicArr})
				return preValue
			else:
				print("No checkpoint file found")
				return -1


# 预处理函数，包括resize、转换灰度图、二值化操作
def pre_pic(picName):
	img = Image.open(picName)
	reIm = img.resize((28,28), Image.ANTIALIAS)
	im_arr = np.array(reIm.convert('L'))
	threshold = 50 #设定合理的阈值，去除噪声
	for i in range(28):
		for j in range(28):
			im_arr[i][j] = 255 - im_arr[i][j]
 			if (im_arr[i][j] < threshold):
 				im_arr[i][j] = 0
			else: im_arr[i][j] = 255

	nm_arr = im_arr.reshape([1, 784])
	nm_arr = nm_arr.astype(np.float32)
	img_ready = np.multiply(nm_arr, 1.0/255.0)

	return img_ready

def application():
	testNum = input("input the number of test pictures:")
	for i in range(testNum):
		testPic = raw_input("the path of test picture:")
		testPicArr = pre_pic(testPic) # 对手写数字图片做预处理
		preValue = restore_model(testPicArr)
		print "The prediction number is:", preValue

def main():
	application()

if __name__ == '__main__':
	main()		
