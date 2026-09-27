#user_id:用户id   order_dt:购买日期
# order_products:购买产品数量  order_amount:购买金额
#数据时间：1997/1-1998/6 用户行为数据约6万条

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
plt.rcParams['font.sans-serif']=['SimHei']
plt.style.use('ggplot') #更改绘图风格

#导入数据
columns=['user_id','order_dt','order_products','order_amount']
df=pd.read_table('CDNOW_master.txt',names=columns,sep='\s+')
#sep:'\s+'匹配任意个空格，源数据列表之间空格数不一致
#日期格式需要转换，存在一个用户一天内购买多次的行为
df.describe()
#用户平均每笔订单购买2.4个产品，标准差2.3，属于正常，然而75%分位数是3，
#说明绝大多数订单购买量不大，围绕在2-3个之间。购买金额反映出大部分订单集中在中小额，30-45左右

#数据预处理
#将日期转换为格式为年月日的时间结构
#%Y：四位的年份  %m:两位月份数字比如05  %d:两位日比如31  %y:两位的年份比如94
#%h:两位小时数09  %M 两位的分钟数  %s 两位的秒数
df['order_date']=pd.to_datetime(df['order_dt'],format='%Y%m%d')

#将order_date转化为精度为月份的数据列
df['month']=df['order_date'].astype('datetime64[M]')

#用户整体消费趋势分析（按月份）
#按月份统计产品购买数量，消费金额，消费次数，消费人数。
plt.figure(figsize=(20,15)) #尺寸大小
#每月的产品购买数量
plt.subplot(221)  #设置两行两列，第一个位置
df.groupby(by='month')['order_products'].sum().plot() #默认折线图
plt.title('每月的产品购买数量')

#每月的消费金额
plt.subplot(222)  #设置两行两列，第二个位置
df.groupby(by='month')['order_amount'].sum().plot() #默认折线图
plt.title('每月的消费金额')

#每月的消费次数
plt.subplot(223)  #设置两行两列，第三个位置
df.groupby(by='month')['user_id'].count().plot() #默认折线图
plt.title('每月的消费次数')

#每月的消费人数（需要对用户id去重）
plt.subplot(224)
df.groupby(by='month')['user_id'].apply(lambda x:len(x.drop_duplicates())).plot() #默认折线图
plt.title('每月的消费人数')

#从上述图一可知前三个月销量非常高，以后销量稳定并且略有下降
#图二每月消费人数在前三个月也比较高，后面较少，可能是春节期间，促销活动
#图三可以看出前三个月订单数量在1w左右，后续月份的平均消费单数2500左右
#图四可以看出前三个月消费人数在8000-10000左右，后续平均消费人数2000以下
#总结：所有数据显示，97年前三个月消费异常，后续趋于常态化


#用户个体消费分析
#1.用户消费金额，消费次数（产品数量）描述统计
user_grouped=df.groupby(by='user_id').sum()
print(user_grouped.describe())
print('用户数量：',len(user_grouped))
#从用户的角度：用户数量23570个，每个用户平均购买7个，平均值大于中位数，属于右偏分布
#消费金额角度来看，平均用户消费106，中位数43，并且存在土豪用户消费13990，
# 结合分位数和最大值来看，平均数与75%分位数几乎相等
#属于典型右偏分布，说明存在小部分用户高额消费

#绘制产品购买量与消费金额散点图
df.plot(kind='scatter',x='order_products',y='order_amount')
#从图中可知，用户的消费金额与购买量呈现线性趋势，每个商品均价在15左右
#订单极值点较少（消费金额大于1000，或者购买量大于60）对于样本影响不大，忽略不计


#分析用户消费分布
plt.figure(figsize=(12,4))
plt.subplot(121)
plt.xlabel('每个订单消费金额')
df['order_amount'].plot(kind='hist',bins=20)
#消费金额在100以内的订单占据了绝大多数

plt.subplot(122)
plt.xlabel('每个uid购买的数量')
df.groupby(by='user_id')['order_products'].sum().plot(kind='hist',bins=50)
#每个用户购买数量很小，集中在50以内，
#两幅图可知，用户主要是消费金额低，并且购买数量小于50的人群（在电商领域是正常现象）

#用户累积消费贡献
#进行用户分组，取出消费金额，进行求和，排序，重置索引
user_cumsum=df.groupby(by='user_id')['order_amount'].sum().sort_values().reset_index()


#每个用户消费金额累加
#举例累加用法
a=[1,2,3,4,5,6,7]
print(np.cumsum(a))  #[1,3,6,10,15,21,28]
user_cumsum['amount_cumsum']=user_cumsum['order_amount'].cumsum()
#消费金额总值
amount_total=user_cumsum['amount_cumsum'].max()
#按照列追加用户累积贡献度prop，前xx名用户贡献率
user_cumsum['prop']=user_cumsum.apply(lambda x:x['amount_cumsum']/amount_total,axis=1)
user_cumsum['prop'].plot() #x轴表示用户数量，y轴表示贡献度，折线图
#图片可知，前2万名用户贡献金额的40%，剩余的用户贡献了60%，（2/8原则）


#用户消费行为
#1.首购时间
#用户分组，取日期最小值即为首购时间
df.groupby(by='user_id')['order_date'].min().value_counts().plot()
#x轴为首次购买日期，y轴为购买频数,由图可知，首次购买的用户量在1月1号-2月10号明显上升，
# 后续开始下降，猜测可能是公司推广力度或者价格调整

#2.最后一次购买日期
df.groupby(by='user_id')['order_date'].max().value_counts().plot()
#x轴为最后购买日期，y轴为购买频数，大多数用户最后购买时间集中在前三个月，说明缺少忠诚客户
#随着时间推移，最后一次购买商品的用户量呈现上升趋势


#构建RFM模型，R；recency最近一次购买时间，越小越好
#F：frequency消费频率越大越好（在本数据表中用购买数量表示频率）
#M：monetary消费金额，越大越好
#透视表的使用（index：相当于groupby，values：取出的数据列，
# aggfunc聚合函数，key必须存在于values列中，并且必须跟有效的聚合函数）
rfm=df.pivot_table(index='user_id',
                   values=['order_date','order_amount','order_products'],
                   aggfunc={
                       'order_date':'max', #最后一次购买
                       'order_products':'sum', #购买总数
                       'order_amonut':'sum'#消费总金额
                   })
rfm.head()

#用户最后一次购买时间与日期列中最大日期的间隔时间，转为天数，保留一位小数
#注意，实际使用要用当前日期才对，举例如下
'''from datetime import datetime
now=datetime.now()
rfm['R']=(now-rfm['order_date'])

current_time=pd.Timestamp.now()
rfm['R']=(current_time-rfm['order_date'])/np.timedelta64(1,'D')
'''
rfm['R']=-(rfm['order_date']-rfm['order_date'].max())/np.timedelta64(1,'D')
rfm.rename(columns={'order_products':'F','order_amount':'M'},inplace=True)

#RFM计算方式：
#每列数据减去数据所在列的平均值，有正有负,根据结果值符号，如果大于0设置为1，否则是0
#RFM可以得出对应三个数字，0和1组成，比如100，111，010等，可以分成不同的客户层次
#rfm['R']-rfm['R'].mean()
def rfm_func(x):
   level= x.apply(lambda x:'1'if x>=1 else '0')
   label=level['R']+level['F']+level['M']
   d={
       '111':'重要价值客户',
       '011':'重要保持客户',
       '101':'重要发展客户',
       '001':'重要挽留客户',
       '110':'一般价值客户',
       '100':'一般发展客户',
       '000':'一般挽留客户',
       '010':'一般保持客户'

   }
   result=d[label]
   return result
rfm['label']=rfm[['R','F','M']].apply(lambda x:x-x.mean()).apply(rfm_func,axis=1)
rfm.head()

#客户分层可视化
for label,grouped in rfm.groupby(by='label'): #这里的grouped其实是dataframe
    #print(label,grouped)
    x=grouped['F']
    y=grouped['R']
    plt.scatter(x,y,label=label)
plt.legend() #显示图例
plt.xlabel('F')
plt.ylabel('R')



#
pivoted_counts=df.pivot_table(
    index='user_id',
    columns='month',
    values='order_dt',
    aggfunc='count'
).fillna(0)  #将na填充为0

#由于浮点数不直观，并且需要转化为是否消费过即可，用0，1表示
df_purchase=pivoted_counts.applymap(lambda x:1 if x>0 else 0)
#apply 作用在dataframe数据中的一行或者一列数据
#applymap作用在dataframe数据中的每一个元素
#map本身是一个series的函数，在df结构中无法使用map函数，map函数作用在series中的每一个元素
#series相当于单行或者单列，dataframe是完整的表格，多行多列

#判断是否是新用户，活跃用户，不活跃用户，回流用户。
def active_status(data):
    pass
























