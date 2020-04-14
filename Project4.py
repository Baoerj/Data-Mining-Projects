import numpy as np
import pandas as pd
from math import log
import matplotlib.pyplot as plt

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


# 获取最佳分割点
def find_best_split_point(X,y):
    best_gain = 0
    best_dimen = -1
    best_point = -1
    H0 = calculate_D(y)
#   遍历每一属性
    for dimen in range(X.shape[1]):
        index_sorted = np.argsort(X[:,dimen])
#       遍历属性中所有分割点
        for i in range(1,len(X)):
            if X[index_sorted[i-1],dimen] != X[index_sorted[i],dimen]:
#               计算划分点
                mid = (X[index_sorted[i-1],dimen]+X[index_sorted[i],dimen])/2
#               划分数据集
                left_x, left_y, right_x, right_y = divide(X, y, dimen, mid)
#                 return(left_y)
#               计算熵值
                H_l = calculate_D(left_y)
                H_r = calculate_D(right_y)
                H_l_r = len(left_y)/len(y)*H_l + len(right_y)/len(y)*H_r
#               计算和比较信息增益，找出最大的信息增益
                if best_gain < (H0 - H_l_r):
                    best_gain = H0 - H_l_r
                    best_dimen = dimen
                    best_point = mid
#   返回最佳分割的类别，类别中的最佳分割点以及该点分割时的信息增益
    return best_dimen, best_point, best_gain
# 计算输入集合中的熵H           
def calculate_D(label):
    H_D = 0
    unique_value = np.unique(label)
    for i in unique_value:
#         print(label.tolist().count(i))
        H_D += -log((label.tolist().count(i)/len(label.tolist())),2)*(label.tolist().count(i)/len(label.tolist()))
#         print(H_D)
    return H_D
# 划分数据集
def divide(data_d, label_d, dimen_d, mid_d):
    return data_d[data_d[:,dimen_d] <= mid_d], label_d[data_d[:,dimen_d] <= mid_d], data_d[data_d[:,dimen_d] > mid_d], label_d[data_d[:,dimen_d] > mid_d]


# 决策树主程序
def DecisionTree(data_D, label_D, minpoints, threshold):
    size = len(label_D)
    purity = 0
    final_label = -1
#   找出数据集中占比最大的类作为该集合的标签，然后计算纯度
    for i in np.unique(label_D):
        if purity < label_D.tolist().count(i)/size:
            purity = round(label_D.tolist().count(i)/size,2)
            final_label = i
#   判断数据集中的纯度或大小是否满足不再分割的条件
    if size <= minpoints or purity >= threshold:
        txtleaf = 'Label:' + str(final_label) + ', Purity:' + str(purity) + ', Size:' + str(size)
        return txtleaf
#   不满足条件，先找最佳分割点
    dimen_D, split_point, gain = find_best_split_point(data_D, label_D)
#   继续划分数据集
    D_l, label_l, D_r, label_r = divide(data_D, label_D, dimen_D, split_point)
#   设定好树的根值
    txtroot = 'Dimen:' + str(dimen_D)  + ', Split:' + str(split_point) + ', Gain:' + str(round(gain,2))
    myTree = {txtroot:{}}
#   递归地调用，直到满足截至条件，将返回的值写入字典中，作为决策树的叶子节点
    myTree[txtroot][0] = DecisionTree(D_l, label_l, minpoints, threshold)
    myTree[txtroot][1] = DecisionTree(D_r, label_r, minpoints, threshold)
    return myTree
    

def plot_node(node_text,centerPt,parentPt,nodeType):
    """
    画节点
    :param node_text:显示的内容
    :param centerPt: 坐标
    :param parentPt: 坐标
    :param nodeType: 边框的种类
    :return:
    """
    create_plot.ax.annotate(node_text,xy=parentPt,xycoords='axes fraction',
                xytext=centerPt,textcoords='axes fraction',
                va="center", ha="center",
                arrowprops=arrow_args,bbox=nodeType)


def get_num_leafs(mytree):
    """
    获取叶节点数目和树的层数
    """
    num_leafs = 0
    first_str = list(mytree.keys())[0]

    second_dict=mytree[first_str]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__=='dict':
            num_leafs +=get_num_leafs(second_dict[key])
        else:
            num_leafs+=1
    return num_leafs


def get_tree_depth(mytree):
    """获取树的深度"""
    max_depth =0
    first_str =list(mytree.keys())[0]
    second_dict = mytree[first_str]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            this_depth =1+ get_tree_depth(second_dict[key])
        else:
            this_depth = 1
        if this_depth>max_depth:
            max_depth=this_depth
    return max_depth


def plot_mid_text(cntrPt,parentPt,txt_sting):
    """画文本 """
    xMid = (parentPt[0]+cntrPt[0])/2.0
    yMid = (parentPt[1]+cntrPt[1])/2.0
    create_plot.ax.text(xMid,yMid,txt_sting)


def plot_tree(myTree,parentPt,nodeTxt):
    """画树"""
    num_leafs=get_num_leafs(myTree)
    depth = get_tree_depth(myTree)

    first_str = list(myTree.keys())[0]
    cntrPt=(plot_tree.x0ff+ (1.0+float(num_leafs))/2.0/plot_tree.totalW,
            plot_tree.y0ff )
    plot_mid_text(cntrPt,parentPt,nodeTxt)
    plot_node(first_str,cntrPt,parentPt,decision_node)
    second_dict = myTree[first_str]
    plot_tree.y0ff = plot_tree.y0ff - 1.0/plot_tree.totalD

    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
           plot_tree(second_dict[key],cntrPt,str(key))
        else:
            plot_tree.x0ff =plot_tree.x0ff +1.0/plot_tree.totalW
            plot_node(second_dict[key],(plot_tree.x0ff,plot_tree.y0ff),
                      cntrPt,leaf_node)

            plot_mid_text((plot_tree.x0ff,plot_tree.y0ff),cntrPt,str(key))

    plot_tree.y0ff = plot_tree.y0ff + 1.0/plot_tree.totalD


def create_plot(inTree):
    fig = plt.figure(figsize=(14,8),facecolor='white')
    fig.clf()
    axprops = dict(xticks=[],yticks=[])
    create_plot.ax=plt.subplot(111,frameon=False,**axprops)
    plot_tree.totalW  = float(get_num_leafs(inTree))
    plot_tree.totalD  = float(get_tree_depth(inTree))
    plot_tree.x0ff = -0.5/plot_tree.totalW
    plot_tree.y0ff =1.0
    plot_tree(inTree,(0.5,1.0),'')
    plt.savefig('output1.jpg')
    plt.show()


if __name__ == '__main__':

    my_Tree = (DecisionTree(nume, lab, 5, 0.95))
    print('用字典结构存储得到的决策树：\n',my_Tree)


    decision_node = dict(boxstyle="sawtooth",fc="0.8")
    leaf_node = dict(boxstyle="round4",fc="0.8")
    # 设置线，这个是带有箭头的线
    arrow_args = dict(arrowstyle="<-")
    create_plot(my_Tree)