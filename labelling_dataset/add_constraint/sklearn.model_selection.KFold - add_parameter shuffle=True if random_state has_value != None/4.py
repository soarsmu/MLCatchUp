import pandas
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

from sklearn.datasets import load_boston
boston = load_boston()


X = boston.data
Y = boston.target
print(X.shape,Y.shape)


# sklearn.model_selection.KFold
# 此外我们可以使用 KFold 类自己构造一个 K 折交叉验证器
# KFold这个类是用来分离数据的。可以讲数据集分割成 k-1 个训练集和 1 个测试集。它有三个参数
# n_splits ： 表示分为几个数据子集
# shuffle ： 要不要打乱数据顺序呢，默认为 False
# random_state ： 打乱数据顺序所用的种子，默认不使用
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)

def bulid_model(model_name):
    model = model_name()
    return model

for model_name in [LinearRegression,Ridge,Lasso,ElasticNet,KNeighborsRegressor,DecisionTreeRegressor,SVR]:
    import time
    # time.clock()是cpu运行时间，time.time()是系统时间
    starttime = time.clock()
    model = bulid_model(model_name)

#  cross_val_score 是用来进行交叉验证的。它有如下参数：
# estimator ： 模型
# X ： 要拟合的数据
# y ： 标签
# cv ： 交叉验证生成器，决定分割验证的策略。有四个可选值： None（默认使用 3 折叠交叉验证），integer（指定 StratifiedKFold 的k值大小），对象，迭代。

    results = model_selection.cross_val_score(model, X, Y, cv=5, scoring='accuracy')
    print(model_name)
    print(results.mean())
    #long running
    endtime = time.clock()
    print('Running time: %s Seconds'%(endtime-starttime))

