import pandas as pd 
import numpy as np

# 获取数据
data = np.fromfile('E:/Data_mining/datasets/iris_dataset.txt',sep=' ').reshape(150,4)
# print(data)

"""第一问"""
#  定义齐次二阶多项式核函数
def _poly(x, y):
    return(x.dot(y.T)) ** 2

# 初始化核矩阵
K = np.zeros([150,150])

# 利用输入空间核函数计算核矩阵
for rowa in range(0,150):
    for rowb in range(0,150):
        K[rowa][rowb] = _poly(data[rowa],data[rowb])
print("利用齐次二阶多项式核函数计算出的核矩阵K：\n" , K)
print("核矩阵K的大小：\n" , K.shape)

# 核矩阵中心化
I = np.identity(150) #创建单位矩阵
# print(I)
singular_mat = np.ones([150,150]) #奇异矩阵
K1 = np.dot(np.dot((I - 1/150 * singular_mat) , K) , (I - 1/150 * singular_mat))
print("中心化后的核矩阵K1：\n" , K1)

# 中心化核矩阵规范化
K2 = np.zeros([150,150])
for rowa in range(0,150):
    for rowb in range(0,150):
         K2[rowa][rowb] = K1[rowa][rowb] / ((K1[rowa][rowa] * K1[rowb][rowb]) ** 0.5)
            
print("中心规范化核矩阵K2：\n" , K2)

"""第二问"""
# 利用齐次二次核将每个点变换到特征空间
data_ = np.zeros([150,10]) # 特征空间中共有10个维度
for rowa in range(0,150):
    data_[rowa][0] = data[rowa][0] ** 2
    data_[rowa][1] = data[rowa][1] ** 2
    data_[rowa][2] = data[rowa][2] ** 2
    data_[rowa][3] = data[rowa][3] ** 2
    data_[rowa][4] = data[rowa][0] * data[rowa][1] * (2 ** 0.5)
    data_[rowa][5] = data[rowa][0] * data[rowa][2] * (2 ** 0.5)
    data_[rowa][6] = data[rowa][0] * data[rowa][3] * (2 ** 0.5)
    data_[rowa][7] = data[rowa][1] * data[rowa][2] * (2 ** 0.5)
    data_[rowa][8] = data[rowa][1] * data[rowa][3] * (2 ** 0.5)
    data_[rowa][9] = data[rowa][2] * data[rowa][3] * (2 ** 0.5)
print("映射到特征空间后的点为：\n" , data_)

# 居中化特征空间内的点
mean = np.array([np.mean(attr) for attr in data_.T]) #特征均值
print('特征均值:\n' ,mean)
center = data_ - mean #中心化
print("居中化特征空间内的点：\n", center)

# 标准化居中后的点
norm = np.zeros([150,150])
for rowa in range(0,150):
    for rowb in range(0,10):
        dima = center[rowa]
        norma = np.linalg.norm(dima)
        norm[rowa][rowb] = center[rowa][rowb]/norma

print("标准化居中后的点：\n",norm)

"""第三问"""
# 特征空间中的中心归一化点的点对积产生的核矩阵
K_= np.zeros([150,150])

for rowa in range(0,150):
    for rowb in range(0,150):
        K_[rowa][rowb] = norm[rowa].dot(norm[rowb].T)
        
print("中心标准化特征空间得到的核矩阵K_：\n",K_)

# 保留两种方法求得的核矩阵的小数点后8位，比较二者是否相等
print(np.around(K2,decimals = 8) == np.around(K_,decimals = 8))