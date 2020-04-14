import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import mpl_toolkits.axisartist as axisartist

# 设置中文字体
my_font = font_manager.FontProperties(fname="C:/Windows/Fonts/AdobeKaitiStd-Regular.otf")
# 设置图形的风格
%matplotlib inline 
%config InlineBackend.figure_format = 'retina'

# 获取数据
data = np.fromfile('E:/Data_mining/new.txt',sep=' ').reshape(19020,10)

"""第一问"""
# 计算均值
mean = np.array([np.mean(attr) for attr in data.T])
print('样本集的特征均值:\n',mean)

# 分别计算两个属性的协方差
def covariance(X, Y):
        num = np.shape(X)[0]
        X, Y = np.array(X), np.array(Y)
        meanX, meanY = np.mean(X), np.mean(Y)
        #按照协方差公式计算协方差，分母一定是n-1
        cov = sum(np.multiply(X-meanX, Y-meanY))/(num-1)
        return cov

"""第二问"""
#特征总数
attrnum = np.shape(data)[1] 
# 初始化协方差矩阵
covmat1 = np.zeros([attrnum,attrnum])
for i in range(attrnum):
    for j in range(attrnum):
        covmat1[i,j] = covariance(data[:,i], data[:,j])
print('按照协方差公式求得的协方差矩阵:\n', covmat1)



"""第三问"""
#样例总数
length = np.shape(data)[0] 
#样本集的中心化
center = data - mean 
print('样本集的中心化(每个元素减去当前维度特征的均值):\n', center)
#求协方差矩阵
covmat2 = np.dot(center.T, center)/(length - 1)
print('按照样本集的中心化求得的协方差矩阵:\n', covmat2)

#利用numpy自带的cov函数求解
print('numpy.cov()计算的协方差矩阵:\n', np.cov(data.T))

"""第四问"""
# 提取第一第二属性
attr1 = center[:,[0]].reshape(19020)
attr2 = center[:,[1]].reshape(19020)
# 计算余弦值
num = np.dot(attr1.T, attr2)
denom = np.linalg.norm(attr1) * np.linalg.norm(attr2)
cos = num / denom
print("\n属性之间余弦值：",cos)

# 绘制散点图
fig = plt.figure(figsize=(16,8),dpi=300,facecolor='white') #图片属性
# 添加坐标轴
ax = axisartist.Subplot(fig, 1,1,1)
fig.add_axes(ax)
ax.axis[:].set_visible(False)
# 设置坐标轴的样式及位置
ax.axis["x"] = ax.new_floating_axis(0, 0)
ax.axis["y"] = ax.new_floating_axis(1, 0)
ax.axis["x"].set_axis_direction('bottom')
ax.axis["y"].set_axis_direction('left')
ax.axis["x"].set_axisline_style("->", size = 2.0)
ax.axis["y"].set_axisline_style("->", size = 2.0)
# 绘制散点
plt.scatter(attr1, attr2, s = 3., c = 'green')
plt.legend(loc="upper right",prop=my_font)
plt.title('Attributes 1 and 2',fontsize = 20, pad = 20, c = 'red')
# 刻度
ax.set_xticks(np.arange(-60,300,20))
ax.set_yticks(np.arange(-40,250,30))
ax.set_xlim(-70, 310)
ax.set_ylim(-60, 240)
# 网格
plt.grid(alpha = 0.2)
plt.show()

"""第五问"""
# 计算均值和标准差
std = attr1.std()
mean1 = attr1.mean(axis=0)
print("属性1的均值为：\n",mean1)
print("属性1的方差为：\n", std**2)

# 构造正态分布的概率密度函数
def normfun(x, mu, sigma):
    pdf = np.exp(-((x - mu) ** 2) / ( 2 * sigma))/(sigma * np.sqrt(2 * np.pi))
    return pdf
x = np.arange(-10,200,0.1)
y = normfun(x, mean1, std)

# 绘制概率密度函数图像
fig = plt.figure(figsize=(14,7),dpi=300,facecolor='white')

ax = axisartist.Subplot(fig, 1,1,1)
fig.add_axes(ax)
ax.axis[:].set_visible(False)

ax.axis["x"] = ax.new_floating_axis(0, 0)
ax.axis["y"] = ax.new_floating_axis(1, 0)
ax.axis["x"].set_axis_direction('bottom')
ax.axis["y"].set_axis_direction('left')
ax.axis["x"].set_axisline_style("->", size = 2.0)
ax.axis["y"].set_axisline_style("->", size = 2.0)

plt.plot(x, y,label = 'Probability Density Function')
plt.axhline(y=normfun(mean1,mean1,std),ls=":",c="gray")#添加水平直线
plt.axvline(x=mean1,ls=":",c="gray")#添加垂直直线
plt.scatter(mean1,normfun(mean1,mean1,std),color = 'red',label = 'Point(x = μ)')
plt.legend(loc="upper right",prop=my_font)
plt.title('The Probability Density Function Of Attribute 1\n(μ= 53.25, σ² = 1794.69)',fontsize = 17, pad = 20, c = 'gray')
ax.set_xticks(np.arange(-7,110,6))
ax.set_yticks(np.arange(-0.001,0.0115,0.001))
ax.set_xlim(-15, 110)
ax.set_ylim(-0.002, 0.0115)
plt.grid(alpha = 0.2)
plt.show()

"""第六问"""
# 计算方差
var = [0]*10
for i in range(0,len(var),1):
    var[i] = np.var(data[:,i])
print('所有属性的方差为:\n',var)
# 获取极值索引
index1 = np.argwhere(var == max(var)) 
index2 = np.argwhere(var == min(var)) 
print('方差最大的属性为：\n',int(index1),'\n方差最小的属性为：\n',int(index2))
print("最大的方差为：\n", max(var))
print("最小的方差为：\n", min(var))

"""第七问"""
# 只保留主对角线以上部分,其余为0
covmat= np.triu(covmat1,1)
# 获取极值索引
index11= np.unravel_index(np.argmax(covmat),covmat.shape)
index22 = np.unravel_index(np.argmin(covmat),covmat.shape)
print('最大协方差的两个属性：\n',index11,'\n最小协方差的两个属性\n',index22)
print("最大的协方差为：\n", covmat[index11])
print("最小的协方差为：\n", covmat[index22])