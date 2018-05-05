# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 22:04:00 2018

@author: MAC
"""

import numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#生成正标签数据
x11= numpy.loadtxt(r"C:\Users\MAC\Desktop\percepton\x11.txt")
x12= numpy.loadtxt(r"C:\Users\MAC\Desktop\percepton\x12.txt")
x11=np.array(x11)
x11=np.array([x11])
x11=np.transpose(x11)
x12=np.array(x12)
x12=np.array([x12])
x12=np.transpose(x12)
y1=np.ones((50,1))
label1=np.ones((1,50))
data1=np.hstack([x11,x12])
data1=np.hstack([data1,y1])
#生成负标签数据
x21= numpy.loadtxt(r"C:\Users\MAC\Desktop\percepton\x21.txt")
x22= numpy.loadtxt(r"C:\Users\MAC\Desktop\percepton\x22.txt")
x21=np.array(x21)
x21=np.array([x21])
x21=np.transpose(x21)
x22=np.array(x22)
x22=np.array([x22])
x22=np.transpose(x22)
y2=-np.ones((50,1))
label2=-np.ones((1,50))
data2=np.hstack([x21,x22])
data2=np.hstack([data2,y2])
#将数据合并
data=np.vstack([data1,data2])
label=np.hstack([label1,label2])
#y=np.concatenate(y1,y2)
print(data)

from sklearn import cross_validation
#将原始数据分成训练数据，和测试数据
train_x,test_x,train_y,test_y=cross_validation.train_test_split(data[:,:-1],data[:,-1],test_size=0.25,
                                                                random_state=0,stratify=data[:,-1])

#初始化权重和偏差

#w=np.zeros(len(data))
#w0=w=np.random.randn(2)
w0=w=[10000,10000]
#print("初始权向量",w)
#w=[0,0]
b0=b=-200
#b0=b=np.random.randn(1)
#print("初始截距",b)                
#设置学习率
a=0.2
#print("学习率",a)
c=0
precision=1
i=0
while 1:
  #  for i in range(len(train_x)):
        #如果某次分类错误，则修改权值和偏差
    if train_y[i]*(np.sum(w*train_x[i])+b)<=0:
        w+=a*train_y[i]*train_x[i]
        b+=a*train_y[i]
        c+=1
        i=0
    else:
        i+=1
    if(i>=(len(train_x))):
        print("iteration finish")
        break


#print("迭代次数：",c)
#print("权向量",w)
print("截距",b)

# 绘图显示
plt.axis([-10, 60, -10, 60])
plt.scatter(data[0:50,0],data[0:50,1],c ="r",label = "postive",s = 60) #画正样本点
plt.scatter(data[50:100,0],data[50:100,1],c = "y",label = "negtive",s =60)     #画负样本点
plt.grid(True)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('myPerceptron')

x1=-10
y1= -(b + w[0] * x1) / w[1]
x2=60
y2 = -(b + w[0] * x2) / w[1]
plt.plot([x1,x2],[y1,y2])
plt.show()
#最后在测试集上检验正确率
count=0
for i in range(len(test_x)):
    if train_y[i]*(np.sum(w*train_x[i])+b)<=0:
        count+=1
precision=(len(test_x)-count)/len(test_x)
print("初始权向量",w0)
print("初始截距",b0)
print("学习率",a)   
print("迭代次数：",c)
print("最终权向量",w)
print("最终截距",b)
print("准确率",precision)