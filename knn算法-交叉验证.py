#导包
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import  accuracy_score

#1.记载鸢尾花数据集
iris_data=load_iris()

#2数据预处理
x_train, x_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2,
                                                    random_state=15)
#3特征工程
transfer=StandardScaler()
x_train=transfer.fit_transform(x_train)
x_test=transfer.transform(x_test)

#4模型训练
estimator=KNeighborsClassifier(n_neighbors=3)
#定义字典记录超参可能出现的情况
param_dict={'n_neighbors':[i for i in range(1,11)]}
#创建GridSearchCV对象，寻找最优超参，使用网格搜索+交叉验证
#参1：要计算的最优超参的模型对象
#参2：该模型的超参可能出现的值
#参3：交叉验证的折数，这里的四折表示：每个超参组合，都会进行4折交叉验证，应该计算4*10=40次
#返回值estimator ：处理后的模型对象
estimator=GridSearchCV(estimator,param_dict,cv=4)
#模型训练
estimator.fit(x_train,y_train)
#打印最优超参
print(f'最优评分：{estimator.best_score_}')
print(f'最优超参组合：{estimator.best_params_}')
print(f'最优估计器对象：{estimator.best_estimator_}')
print(f'具体的交叉验证结果：{estimator.cv_results_}')

