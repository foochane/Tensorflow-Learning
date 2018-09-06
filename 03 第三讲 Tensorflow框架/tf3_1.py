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


