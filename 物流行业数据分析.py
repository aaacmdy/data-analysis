#数据清洗
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
plt.rcParams['font.sans-serif']='SimHei'  #设置中文显示

data=pd.read_csv('data_wuliu.csv',encoding='gbk')
data.info()
#通过info可以看出，包括10列数据，名字，数据量，格式等，可以得出：
# 1.订单号，货品交货情况，数量：存在缺失值，但是缺失数量不大，可以删除
# 2.订单行，对分析无关紧要，可以考虑删除
# 3.销售金额格式不对，（万元，元，逗号问题），数据类型不匹配（转为int,float）

#去重，只保留第一次出现的数据，修改源数据
data.drop_duplicates(keep='first',inplace=True)
print(data.info())

#删除缺失值
data.dropna(axis=0,how='any',inplace=True) #any表示只要有一个na都会删除整行，对应还有参数all

#删除订单行那一列数据
data.drop(columns=['订单行'],inplace=True,axis=1)

#更新索引（刚刚删除的行数据导致部分索引缺失不连续）
#drop=true表示将原来的索引删除
data=data.reset_index(drop=True,inplace=True)

#取出销售金额列，对每个数据进行清洗
#编写自定义过滤函数，删除逗号，转成float，如果是万元，则乘以10000，否则，删除元
def data_deal(number):
    if '万元' in number:
        return float(number[:number.find('万元')].replace(',',''))*10000
    else:
        return float(number.replace('元','').replace(',',''))
data['销售金额']=data['销售金额'].map(data_deal)

#异常值处理（销售金额=0，偏态分布）
data=data[data['销售金额']!=0]  #只保留金额不为0的数据
#数量与金额存在右偏现象，但是在电商行业很正常，无需处理


#增加月份数据
data['销售时间']=pd.to_datetime(data['销售时间'])
data['月份']=data['销售时间'].apply(lambda x:x.month)
#apply适用多种数据遍历，map只用于单列数据，上述代码可以使用.dt用法
# data['销售时间']=pd.to_datetime(data['销售时间'])
# data['月份']=data['销售时间'].dt.month

#配送服务是否存在问题
#月份维度
data['货品交货状况']=data['货品交货状况'].str.strip() #strip可以删除前后空格
data1=data.groupby(['月份','货品交货状况']).size().unstack()
#多个分组，size可以计算分组后的统计行数,unstack可以将数据用列显示

data1['按时交货率']=data1['按时交货']/(data1['按时交货']+data1['晚交货'])
#从按时间交货率来看，第四季度低于第三季度，猜测可能是气候原因造成

#销售区域维度
data2=data.groupby(['销售区域','货品交货状况']).size().unstack()
data2['按时交货率']=data2['按时交货']/(data2['按时交货']+data2['晚交货'])
print(data2.sort_values(by='按时交货率',ascending=False))
#西北地区存在突出的延时交货问题，需要解决

#货品维度
data3=data.groupby(['货品','货品交货状况']).size().unstack()
data3['按时交货率']=data3['按时交货']/(data3['按时交货']+data3['晚交货'])
print(data3.sort_values(by='按时交货率',ascending=False))
#货品四晚交货情况严重，需要解决

#货品和销售区域结合
data4=data.groupby(['货品','销售区域','货品交货状况']).size().unstack()
data4['按时交货率']=data4['按时交货']/(data3['按时交货']+data4['晚交货'])
print(data4.sort_values(by='按时交货率',ascending=False))
#销售区域：最差在西北地区，货品有1和4，主要是货品4送货较晚
#货品：最差的货品2，主要送往华东和马来西亚，主要是马来西亚送货晚导致。

#是否存在尚有潜力的销售区域
#a.月份维度
data5=data.groupby(['月份','货品'])['数量'].sum().unstack()
data5.plot(kind='line') #绘制折线图数据可视化
#货品2在10月和12月份，销量猛增，
# 原因猜测有二：1.公司加大营销力度 2.开发了新的市场

#b不同区域
data6=data.groupby(['销售区域','货品'])['数量'].sum().unstack()
data6.plot(kind='line') #绘制折线图数据可视化
#从销售区域看，每种货品销售区域为1-3个，货品1有三个销售区域，
# 货品2有两个销售区域，其余货品均有1个销售区域

#c月份和区域
data7=data.group(['月份','销售区域','货品'])['数量'].sum().unstack()
data7['货品2']
#货品2在10月和12月销量猛增，原因主要发生在原有的销售区域（华东）
#同样，分析出在7，8，9，11月份的销售数量还有很大的提升空间，可以适当加大营销力度


#商品是否存在质量问题
data['货品用户反馈']=data['货品用户反馈'].str.strip() #取出空格
data8=data.groupby(['货品','销售区域'])['货品反馈情况'].value_counts().unstack()
data8['拒货率']=data8['拒货']/data8.sum(axis=1)
data8['返修率']=data8['返修']/data8.sum(axis=1)
data8['合格率']=data8['质量合格']/data8.sum(axis=1)
data8.sort_values(['合格率','返修率','拒货率'],ascending=False)
#货品3，5，6合格率较高，返修率较低，说明质量还可以
#货品1，2，4合格率较低，返修率较高，说明质量存在一定的问题，需要改善
#货品2在马来西亚拒货率很高，同时，在马来西亚按时交货率也很低
#猜测马来西亚人对送货时效性要求高，如果达不到，则往往考虑拒货
#考虑到货品2在华东地区销售量大，可以适当考虑增加在华东的投资，减少马来西亚的投入













