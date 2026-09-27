'''案例：演示knn算法识别图片，手写数字识别案例
 介绍：每张图片都是由28*28像素组成，即：
 我们的csv文件每一行有784个像素点，表示图片每个像素的颜色，标签列是具体的真实数字'''

import  matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib #保存模型用的模块
from collections import Counter

#1.定义函数，接收用户传入的索引，展示，该索引对应的图片
def show_digit(idx):
    #1.读取数据集，获取数据源
    df=pd.read_csv('./data/手写数字识别.csv')

    #2.判断传入索引是否越界
    if idx<0 or idx>len(df)-1:
        print('索引越界')
        return

    #3.正常情况，正常获取数据
    x=df.iloc[:,1:] #全部行，第2列
    y=df.iloc[:,0]  #全部行，第1列

    #4.查看用户传入的索引对应的图片是多少
    print(f'该图片对应的数字是：{y.iloc[idx]}')
    print(f'查看所有图片的标签分布情况：{Counter(y)}')
    #5查看图片的形状
    print(x.iloc[idx].shape) # (784,)
    print(x.iloc[idx].values)  #具体的784个像素点数据

    #6把（784，）转成（28，28）
    x=x.iloc[idx].values.reshape(28,28)

    #7具体绘制图的动作
    plt.imshow(x,camp='gray')
    plt.axis('off')  #不显示坐标
    plt.show()

#2定义函数训练模型，并保存训练好的模型
def train_model():
    pass
