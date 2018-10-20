import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.datasets import load_boston

#获取 Boston 房价数据集，共有506条数据，每条数据有13个特征，说明如下：
# CRIM：城镇人均犯罪率。 
# ZN：住宅用地超过 25000 sq.ft. 的比例。 
# INDUS：城镇非零售商用土地的比例。 
# CHAS：查理斯河空变量（如果边界是河流，则为1；否则为0）。 
# NOX：一氧化氮浓度。 
# RM：住宅平均房间数。 
# AGE：1940 年之前建成的自用房屋比例。 
# DIS：到波士顿五个中心区域的加权距离。 
# RAD：辐射性公路的接近指数。 
# TAX：每 10000 美元的全值财产税率。 
# PTRATIO：城镇师生比例。 
# B：1000（Bk-0.63）^ 2，其中 Bk 指代城镇中黑人的比例。 
# LSTAT：人口中地位低下者的比例。 
# MEDV：自住房的平均房价，以千美元计
boston,prices = load_boston(True)
rm = [i[5] for i in boston]     #以住宅平均房间数为特征，求出房价和住宅平均房间数之间的线性回归方程

#划分训练集和测试集，比例为8：2
ratio = 0.8
offset = int(len(rm)*0.8)
x_input = rm[:offset]
y_input = prices[:offset]
x_test = rm[offset:]
y_test = prices[offset:]

#构建线性回归模型
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0)) #初始化weight
b = tf.Variable(tf.zeros([1])) #初始化bias
y = W*x_input + b #构建线性回归方程

#定义loss function，即计算((y(i) - y) ^ 2)/N
loss = tf.reduce_mean(tf.square(y - y_input))

#使用梯度下降法优化loss函数
optimizer = tf.train.GradientDescentOptimizer(0.005) #设置学习率为0.005
train = optimizer.minimize(loss)

#创建会话并初始化变量
session = tf.Session()
init = tf.global_variables_initializer()
session.run(init)

#开始训练
for step in range(len(x_input)):
    session.run(train)
    if step % 20 == 0:    #每隔20步打印一下调优状况
        print("Step=%d, Loss=%f, [Weight=%f Bias=%f]"%(step, session.run(loss), session.run(W), session.run(b)))
    
#绘制回归直线
plt.plot(x_input, y_input, '.', label='training set')
plt.title("house price prediction of boston")
plt.plot(x_input, session.run(W)*x_input+session.run(b), 'r', label='Fitted Line')
plt.legend()
plt.xlabel('rm')
plt.ylabel('price')
plt.show()

#绘制对test数据集的拟合程度
plt.plot(x_test, y_test, '.', label='test set')
plt.title("house price prediction of boston")
plt.plot(x_input, session.run(W)*x_input+session.run(b), 'r', label='Fitted Line')
plt.legend()
plt.xlabel('rm')
plt.ylabel('price')
plt.show()

#关闭会话
session.close()
