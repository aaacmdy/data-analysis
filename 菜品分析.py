import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']='SimHei'  #设置中文显示

#1.加载数据
data1=pd.read_excel('meal_order_detail.xlsx',sheet_name='meal_order_detail1')
data2=pd.read_excel('meal_order_detail.xlsx',sheet_name='meal_order_detail2')
data3=pd.read_excel('meal_order_detail.xlsx',sheet_name='meal_order_detail3')

#2.数据预处理（合并数据，NA等处理，分析数据
data=pd.concat([data1,data2,data3],axis=0)  #按照行进行拼接
data.dropna(axis=1,inplace=True)  #按照列删除na,并且修改源数据，如果不加true，默认存放副本
data.info()

#3.数据可视化matplotlib
#统计卖出菜品的价格，两种方法
round(data['amounts'].mean(),2)
round(np.mean(data['amounts']),2)

#频数统计，什么菜最受欢迎
dishes_count=data['dishes_name'].value_counts()[:10]  #取前十条

dishes_count.plot(kind='line',color=['r']) #红色折线
dishes_count.plot(kind='bar',fontsize=16) #设置16号字体大小

for x,y in enumerate(dishes_count):
    print(x,y)
    plt.text(x,y+2,y,ha='center',fontsize=12)
    #x表示索引位置，第一个y表示高度，第二个y表示显示的内容,ha表示位置，左中右

#订单点菜的种类最多
data_group=data['order_id'].value_counts()[:10]
data_group.plot(kind='bar',fontsize=16,color=['r','o','y','g','b','p'])
plt.title('订单点菜种类top10')
plt.xlabel('订单id')
plt.ylabel('点菜种类')
plt.show()
#八月份餐厅订单点菜种类前十名，平均点菜25个菜品（可以根据里面的菜品，加大营销力度）

#订单id点菜数量前十名（分组order-id,counts求和，排序，前十）
data['total_amounts']=data['counts']*data['amounts']  #统计单个菜品消费总额
dataGroup=data[['order_id','counts','amounts','total_amounts']].groupby(by='order_id')
#写一个新的表格，有四列组成，按照订单号分组
Group_sum=dataGroup.sum()
sort_counts=Group_sum.sort_values(by='counts',ascending=False)
sort_counts['counts'][:10].plot(kind='bar',fontsize=16)

#哪个订单id吃的钱最多
sort_total_amounts=Group_sum.sort_values(by='total_amounts',ascending=False)
sort_total_amounts['total_amounts'][:10].plot(kind='bar',fontsize=16)
plt.xlabel('订单id')
plt.ylabel('消费金额')
plt.title('消费金额前十')
plt.show()

#哪个订单平均消费最贵
Group_sum['avg_expense']=Group_sum['total_amounts']/Group_sum['counts']
sort_avg_expense=Group_sum.sort_values(by='avg_expense',ascending=False)
sort_avg_expense['avg_expense'][:10].plot(kind='bar',fontsize=16)
plt.xlabel('订单id')
plt.ylabel('消费平均金额')
plt.title('消费平均金额前十')
plt.show()


#一天当中什么时段点菜最多
data['hourcount']=1
data['time']=pd.to_datetime(data['place_order_time'])
data['hour']=data['time'].map(lambda x: x.hour) #使用函数将时间提取出小时数
gp_by_hour=data.groupby(by='hour').count()['hourcount']
#根据hour分组，只显示hourcount的计数情况
gp_by_hour.plot(kind='bar')
plt.xlabel('时间')
plt.ylabel('订单数')
plt.title('每小时点菜订单数量')
plt.show()

#哪一天的订单数量最多
data['daycount']=1
data['day']=data['time'].map(lambda x :x.day)
gp_by_day=data.groupby(by='day').count()['daycount']
gp_by_day.plot(kind='bar')
plt.xlabel('日期')
plt.ylabel('订单数')
plt.title('每天点菜订单数量')
plt.show()
#最高值出现的前五天
gp_by_day_top5=gp_by_day.sort_values(ascending=False).head(5)

#查看星期几人数最多，订餐数最多，映射到星期
data['weekcount']=1
data['weekday']=data['time'].map(lambda x:x.weekday())
week_map={
    0:'星期一',
    1:'星期二',
    2:'星期三',
    3:'星期四',
    4:'星期五',
    5:'星期六',
    6:'星期日'
}
data['weekday_cn']=data['weekday'].map(week_map)
gp_by_weekday=data.groupby(by='weekday_cn').count()['weekcount']
gp_by_day.plot(kind='bar')
plt.xlabel('星期')
plt.ylabel('订单数')
plt.title('星期数点菜订单数量')
plt.show()





















