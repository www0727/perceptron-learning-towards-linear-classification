# -*- coding: utf-8 -*-
"""
Created on Tue May  1 00:38:40 2018

@author: MAC
"""
import numpy
import pandas as pd
import numpy as np
#生成正标签数据
x11=5*np.random.randn(50,1)+40
x12=5*np.random.randn(50,1)+40
y1=np.ones((50,1))
#data1=np.hstack([x1,y1])
#生成负标签数据
x21=5*np.random.randn(50,1)+10
x22=5*np.random.randn(50,1)+30
y2=-np.ones((50,1))
#data2=np.hstack([x2,y2])
#将数据合并
#data=np.vstack([data1,data2])
#print(x11.shape)
numpy.savetxt(r"C:\Users\MAC\Desktop\percepton\x11.txt",x11)
numpy.savetxt(r"C:\Users\MAC\Desktop\percepton\x12.txt",x12)
numpy.savetxt(r"C:\Users\MAC\Desktop\percepton\x21.txt",x21)
numpy.savetxt(r"C:\Users\MAC\Desktop\percepton\x22.txt",x22)