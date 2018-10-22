# -*- coding: utf-8 -*

"""
data importion
"""
import numpy as np
import matplotlib.pyplot as plt  

# 载入CSV文件
dataset = np.loadtxt('C:/Users/DELL/Desktop/data/watermelon3.0a.csv', delimiter=",")

# 从目标属性分离数据
X = dataset[:,1:3]
y = dataset[:,3]

m,n = np.shape(X)  

# 绘制散点图显示原始数据
f1 = plt.figure(1)       
plt.title("watermelon3.0a")
plt.xlabel("密度")  
plt.ylabel("含糖率")  
plt.scatter(X[y == 0,0], X[y == 0,1], marker = "o", color = "k", s=100, label = "bad")
plt.scatter(X[y == 1,0], X[y == 1,1], marker = "o", color = "r", s=100, label = "good")
plt.legend(loc = 'upper right')
#plt.show()

"""
使用sklearn lib实现logistic回归
"""
import matplotlib.pylab as pl
from sklearn import metrics
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.5, random_state=0)
# 训练模型
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

# 确认模型
y_pred = log_model.predict(X_test)

# 总结模型的拟合性
print(metrics.confusion_matrix(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))

precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_pred)    

# 显示plt的决策边界
f2 = plt.figure(2) 
h = 0.001
x0_min, x0_max = X[:, 0].min()-0.1, X[:, 0].max()+0.1
x1_min, x1_max = X[:, 1].min()-0.1, X[:, 1].max()+0.1
x0, x1 = np.meshgrid(np.arange(x0_min, x0_max, h),np.arange(x1_min, x1_max, h))

# 模型的预测（分类）函数
z = log_model.predict(np.c_[x0.ravel(), x1.ravel()]) 

# 彩色图显示结果
z = z.reshape(x0.shape)
plt.contourf(x0, x1, z, cmap = pl.cm.Paired )

plt.title("watermelon3.0a")
plt.xlabel("密度")  
plt.ylabel("含糖率")  
plt.scatter(X[y == 0,0], X[y == 0,1], marker = "o", color = "k", s=100, label = "bad")
plt.scatter(X[y == 1,0], X[y == 1,1], marker = "o", color = "r", s=100, label = "good")
# plt.show()

"""
实现logistic回归的代码
"""
from sklearn import model_selection
import self_def

# X_train, X_test, y_train, y_test
np.ones(n)
m,n = np.shape(X)
X_ex = np.c_[X, np.ones(m)]
X_train, X_test, y_train, y_test = model_selection.train_test_split(X_ex, y, test_size=0.5, random_state=0)


# 使用梯度下降法求最优参数beta = [w, b]
beta = self_def.gradDscent_2(X_train, y_train)

# beta映射到模型
y_pred = self_def.predict(X_test, beta)
m_test = np.shape(X_test)[0]

# 混淆矩阵的计算与预测精度
cfmat = np.zeros((2,2))
for i in range(m_test):   
    if y_pred[i] == y_test[i] == 0: cfmat[0,0] += 1 
    elif y_pred[i] == y_test[i] == 1: cfmat[1,1] += 1 
    elif y_pred[i] == 0: cfmat[1,0] += 1 
    elif y_pred[i] == 1: cfmat[0,1] += 1 
                                
print(cfmat)
