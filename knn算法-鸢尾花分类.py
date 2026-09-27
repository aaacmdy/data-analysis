'''
案例：通过KNN算法实现鸢尾花的分类操作
 问题：机器学习研发流程
 1.加载数据
 2.数据预处理
 3.特征工程（提取，预处理...)
 4.特征预处理
 5.模型评估
 6.模型预测
'''
#导包
from sklearn.datasets import load_iris  #加载鸢尾花数据
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split #分割训练集与数据集
from sklearn.preprocessing import StandardScaler  #数据标准化
from sklearn.neighbors import KNeighborsClassifier #KNN算法，分类对象
from sklearn.metrics import accuracy_score #模型评估，计算模型准确性

#1.定义函数，加载鸢尾花数据
def dm01_loadiris():
    #1.加载鸢尾花数据集
    iris_data=load_iris()
    #2.查看数据
    print(f'数据集：{iris_data}')
    print(f'数据集的类型：{type(iris_data)}')
    #查看数据集的所有键
    print(f'数据所有的键：{iris_data.keys()}')
    #查看数据集对应的值
    print(f'具体的数据：{iris_data.data[:5]}') #查看前五行数据
    print(f'查看标签：{iris_data.target[:5]}')
    print(f'查看标签名称：{iris_data.target_names}')
    print(f'查看特征对应的名称：{iris_data.feature_names}')
    print(f'数据的描述：{iris_data.DESCR}')
    print(f'数据的框架：{iris_data.frame}')
    print(f'数据集的文件名：{iris_data.filename}')
    print(f'数据集的模型（在哪个包下）：{iris_data.data_module}')

#2.定义函数，绘制数据集的散点图
def dm02_show_iris():
    #1.加载数据集
    iris_data=load_iris()
    #2.那鸢尾花数据集封装成dataframe对象。
    iris_df=pd.DataFrame(iris_data.data,columns=iris_data.feature_names)
    #3给df对象新增1列标签列
    iris_df['label']=iris_data.target
    print(iris_df)
    #4.通过seaborn绘制散点图
    #参1：数据集 参2：x轴 参3：y轴 参4：分组字段 参5：是否显示拟合线
    sns.lmplot(data=iris_df,x='sepal length(cm)',y='sepal width (cm)',hue='label',fit_reg=False)
    #5.。设置标题，显示
    plt.title('iris data')
    plt.tight_layout() #自动调整子图参数，使整个图像边界与子图匹配
    plt.show()

#3.定义函数，切分训练集和测试集。
def dm03_split_train_test():
    #1.加载数据集
    iris_data=load_iris()
    #2数据预处理，将150个数据按照8：2切分成训练集和测试集
    #参1：特征数据 参2：标签数据 参3：测试集的比例 参4：随机种子，每次生成的数据定格不变
    #返回值：训练集的特征数据，测试集的特征数据，训练集的标签数据，测试集的标签数据
    x_train,x_test,y_train,y_test=train_test_split(iris_data.data,iris_data.target,test_size=0.2,random_state=15)

    #3打印切割后的结果。
    print(f'训练集的特征：{x_train},个数：{len(x_train)}') #120条，每条四列
    print(f'训练集的标签：{y_train},个数：{len(y_train)}') #120条，每条1列
    print(f'测试集的特征：{x_test},个数：{len(x_test)}')  #30条，每条4列
    print(f'测试集的标签：{y_test},个数：{len(y_test)}')  #30条，每条1列

#4.定义函数，实现鸢尾花完整案例，加载数据，数据预处理，特征工程，模型训练，模型评估，模型预测
def dm04_iris_evaluate_test():
    #1加载数据集
    iris_data=load_iris()
    #2数据预处理，按照8：2的比例，切分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2,
                                                        random_state=15)

    #3.特征工程（提取，预处理。。。
    #3.1创建标准化对象
    transfer=StandardScaler()
    #3.2对特征数据进行标准化，
    # fit_transform用于首次标准化，在x训练集数据中使用（会给出均值和方差，后面测试集也是用该均值方差标准化
    x_train=transfer.fit_transform(x_train)
    x_test=transfer.transform(x_test)


    #4模型训练
    #4.1创建模型对象
    estimator=KNeighborsClassifier(n_neighbors=3)
    #4.2具体的训练模型动作
    estimator.fit(x_train,y_train)
    #5模型评估
    #场景1对刚才切分的测试集进行测试
    y_pre=estimator.predict(x_test)
    print(f'预测值为：{y_pre}')
    #场景2对新的数据（150条数据之外）进行测试
    #自定义新数据集
    my_data=[[7.8,2.1,3.9,1.6]]
    my_data=transfer.transform(my_data)
    y_pre_new=estimator.predict(my_data)
    print(f'预测值为：{y_pre_new}')

    #查看上述数据集每种分类预测的可能性
    y_pre_proba=estimator.predict_proba(my_data)
    print(f'各分类预测概率为：{y_pre_proba}')

    #6模型预测
    #方式1直接评分基于：训练集的特征和训练集的标签
    print(f'正确率：{estimator.score(x_train,y_train)}')
    #方式2基于测试集的标签和预测结果进行评分
    print(f'正确率：{accuracy_score(y_test,y_pre)}')



#5.测试
if __name__=='__main__':
    # dm01_loadiris()
    # dm02_show_iris()
    # dm03_split_train_test()
    dm04_iris_evaluate_test()