#coding:utf-8
import tensorflow as tf
a=tf.constant([1.0,2.0])
b=tf.constant([3.0,4.0])
result=a+b
print result

"""
输出的结果为：
Tensor("add:0", shape=(2,), dtype=float32)

add ：节点名
shape :维度信息，括号里只有一个数“2“，表示维度是1且一个维度里有两个元素
dtpye :数据类型
"""

# 测试
c=tf.constant([1.0,2.0])  #Tensor("Const_2:0", shape=(2,), dtype=float32) 是个向量，有两个元素
print c
d=tf.constant([[1.0,2.0]])  #Tensor("Const_3:0", shape=(1, 2), dtype=float32) 1行2列矩阵
print d
e=tf.constant([[1.0],[2.0]])  #Tensor("Const_4:0", shape=(2, 1), dtype=float32) 2行1列矩阵
print e

