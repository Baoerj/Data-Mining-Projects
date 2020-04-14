import pandas as pd
from math import log
import numpy as np
import math

#通过read_table来读取目的数据集
data = pd.read_csv("E:/Data_mining/datasets/iris.txt",delimiter=',',header=None)
#将pandas中的dataframe转换为numpy中的array 
nume = data.ix[:,:3].values
lab = data.ix[:,4].values

# 替换数据，将标签类用数字代替
unique_label = np.unique(lab)
for i in range(0,len(unique_label),1):
    for j in range(0,len(lab),1):
        if lab[j] == unique_label[i]:
            lab[j] = i + 1
    print(i + 1, ':', unique_label[i])

# DENCLUE主函数
def Denclue(D, h, mindensity, E):
    Rx = {}
    A = np.empty(shape=[0,4],dtype = float)
    All_d_a = np.empty(shape=[0,4],dtype = float)
    for x in range(0,len(D),1):
#         寻找密度吸引子
        x_star = Findattractor(D[x], D, mindensity, E, h)
        All_d_a = np.append(All_d_a, [x_star], axis = 0)
    #  判断是否满足密度阈值
    for i in range(0,len(All_d_a),1):
#         print(densityvalue(All_d_a[i], h, D))
        if densityvalue(All_d_a[i], h, D) >= mindensity:
            A = np.append(A, [All_d_a[i]], axis = 0)
#           将密度吸引子所吸引的点存入集合中
            if str(All_d_a[i]) in Rx.keys():
                Rx[str(All_d_a[i])].append(i)
            else:
                Rx.setdefault(str(All_d_a[i]),[]).append(i)

#   计算两两之间密度吸引子是否密度可达，然后聚类
    new_nume = D
    for i in Rx.keys():
        lis = Rx[i]
        new_nume = np.append(new_nume,[All_d_a[lis[0]]],axis=0)
#     print(new_nume)
    cluster_point = {}
    gather, subset = cluster(new_nume,h,mindensity,Rx)
    denrea_ = {}

#   整理密度吸引子集合、类簇点的集合，计算尺寸和纯度
    c = 0
    for point in gather.keys():
        denrea_.setdefault(point,[])
        listA = gather[point]
        cluster_point.setdefault(str(listA),[])
        for i in listA:
            point_list = subset[i]
            for o in range(0,len(point_list),1):
                if point_list[o] < len(D):
                    column = int(point_list[o])
                    denrea_[point].extend(str(lab[column]))
                    cluster_point[str(listA)].append(str(D[column]))
#   计算纯度和获取类标签
    mainlabel, eachpurity = calculate_purity(denrea_)
#   打印出所有数据
    print('The number of clusters:',len(gather))
    for pt in cluster_point.keys():
        c += 1
        print('\n Cluster',c,'\n Density Attractor:',pt,'\n Size:',len(cluster_point[pt]),'\n Label:', mainlabel[c-1],'\n Purity:',eachpurity[c-1], '\n Point set:',cluster_point[pt])

#     寻找密度吸引子
def Findattractor(x, D, mindensity, E, h):
    xt_plus = x
#     递归计算满足条件的密度吸引子
    while True:
        xt = xt_plus
        Knn = KNN(xt, D)
#         在K个邻居点中计算迭代点
        xt_plus = calculate_xt_plus(xt, Knn, h)
        if np.linalg.norm(xt_plus-xt) <= E:
            break
    return xt_plus

#    计算K最近邻点     
def KNN(x,D):
    distance = []
    knnpoints = np.empty(shape=[0,4],dtype = float)
    k = 48
#     求所有点到x的二范数
    for i in range(0,len(D),1):
        distance.append(np.linalg.norm(D[i]-x))
#   获取排序后的索引
    indexknn = np.argsort(distance)
#     提取前K个最近邻点
    for i in range(0,k,1):
        knnpoints = np.append(knnpoints, [D[indexknn[i+1]]], axis=0)
    return knnpoints

# 计算迭代式
def calculate_xt_plus(xt, Knn, h):
    top = [0,0,0,0]
    bottom = 0
    for i in Knn:
        K = gauss(xt, i, h)
        top += i * K
        bottom += K
    return top/bottom
# 计算高斯核
def gauss(xt, xi, h):
    d = len(xt)
    return (1/(2*math.pi)**(d/2))*np.exp(-(np.dot((xt-xi).T, (xt-xi)))/(2*(h**2)))

# 计算密度吸引子的密度
def densityvalue(xstar, h, D):
    f_xstar = 0
#   初始化窗口内的点
    windowpoint = np.empty(shape=[0,4],dtype = float)
    for i in range(0, len(D), 1):
#       判断是否属于该窗口内，然后添加入窗口集合中
        if abs(D[i][0]-xstar[0]) <= h/2 and abs(D[i][1]-xstar[1]) <= h/2 and abs(D[i][2]-xstar[2]) <= h/2 and abs(D[i][3]-xstar[3]) <= h/2:
            windowpoint = np.append(windowpoint,[D[i]],axis=0)
#   计算密度吸引子密度
    for i in windowpoint:
        f_xstar += gauss(xstar, i, h)
    return f_xstar/(len(windowpoint)*(h**2))

# 将密度可达的密度吸引子的被吸引点存入一类中，形成聚类
def cluster(data_, h, mindensity,Rx):
    denrea = {}
    clusters = {}
    finalcluster = {}
    meaningpoint = np.empty(shape=[0,4],dtype = float)
#   计算可达点
    for key in Rx.keys():
        listindex = Rx[key]
        for i in listindex:
            denrea.setdefault(str(key),[])
            denrea[key].extend(densityreachable(data_[i],h,data_,mindensity))
#   判断密度吸引子是否密度可达
    for ke in denrea.keys():
        clusters.setdefault(str(ke),[])
        list1 = np.unique(sorted(denrea[ke]))
#         print(ke, np.unique(sorted(denrea[ke])))
        for key in denrea.keys():
            list2 = np.unique(sorted(denrea[key]))
            if ke != key:
                for i in list1:
                    if i in list2:
                        clusters[ke].append(key)
                        break
#       将密度可达的吸引子和吸引的点存入相应的字典中
        clusters[ke].append(ke)
        denrea[ke] = sorted(np.unique(denrea[ke]))

    List = []
    for l in clusters.keys():
        clusters[l] = sorted(clusters[l])
        List.append(str(l))
#   初始化最终分类簇
    finalcluster.setdefault(List[0],[])
    finalcluster[List[0]] = clusters[List[0]]
#  剔除重复的集合和非最大子集的聚类
    for k in range(0,len(List),1):
        c = 0
        for p in finalcluster.keys():
            if List[k] == p or clusters[List[k]] == finalcluster[p] or set(clusters[List[k]]).issubset(finalcluster[p]):
                c = 1
            elif set(finalcluster[p]).issubset(clusters[List[k]]):
                finalcluster[p] = clusters[List[k]]
                c = 1
        if c == 0:
            finalcluster.setdefault(List[k],[])
            finalcluster[List[k]] = clusters[List[k]]
                
    return finalcluster, denrea
        
# 计算密度可达的点 
def densityreachable(xstar, h, D, mindensity):
    f_xstar = 0
    windowpoint = []
    for i in range(0, len(D), 1):
        f_xstar += gauss(xstar, D[i], h)
        if abs(D[i][0]-xstar[0]) <= h/2 and abs(D[i][1]-xstar[1]) <= h/2 and abs(D[i][2]-xstar[2]) <= h/2 and abs(D[i][3]-xstar[3]) <= h/2 and f_xstar/(len(windowpoint)*(h**2)) >= mindensity:
            windowpoint = np.append(windowpoint,int(i))     
    return windowpoint

# 计算类簇中的纯度，寻找类簇标签
def calculate_purity(labelindex):
    valuable = [0]*3
    purity = [0]*len(labelindex)
    labelcluster = [0]*len(labelindex)
    w = 0
    for ind in labelindex.keys():
        valuable = [0,0,0]
        lt = labelindex[ind]
        for m in range(0,len(lt),1):
            if lt[m] == str(1):
                valuable[0] += 1
            elif lt[m] == str(2):
                valuable[1] += 1
            elif lt[m] == str(3):
                valuable[2] += 1
        maxindex = valuable.index(max(valuable))
        labelcluster[w] = unique_label[maxindex]
        purity[w] = max(valuable)/len(lt)
        w += 1
    return labelcluster,purity

    
np.set_printoptions(precision=2)
Denclue(nume,0.39,0.133375,0.0001)