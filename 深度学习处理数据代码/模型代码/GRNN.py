# -*- coding: utf-8 -*-
# 适用数据量少，广义回归神经网络
import numpy as np
import math
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import data


def distance(X, Y):  # 计算两个样本之间的距离
    return np.sqrt(np.sum(np.square(X - Y), axis=0))

def distance_mat(trainX, testX):  # 计算待测试样本与所有训练样本的欧式距离
    m, n = np.shape(trainX)
    p = np.shape(testX)[0]
    Eu_dis = np.mat(np.zeros((p, m)))
    for i in range(p):
        for j in range(m):
            Eu_dis[i, j] = distance(testX[i, :], trainX[j, :])
    return Eu_dis

def Gauss(Eu_dis, sigma):  # 测试样本与训练样本的距离矩阵对应的Gauss矩阵
    m, n = np.shape(Eu_dis)
    gauss = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            gauss[i, j] = math.exp(- Eu_dis[i, j] / (2 * (sigma ** 2)))
    return gauss

def sum_layer(gauss, trY):  # 求和层矩阵
    m, l = np.shape(gauss)
    n = np.shape(trY)[1]
    sum_mat = np.zeros((m, n + 1))
    # 对所有模式层神经元输出进行算术求和
    for i in range(m):
        sum_mat[i, 0] = np.sum(gauss[i, :], axis=0)  # 第0列为每个测试样本Gauss数值之和
    # 对所有模式层神经元进行加权求和
    for i in range(m):
        for j in range(n):
            total = 0.0
            for s in range(l):
                total += gauss[i, s] * trY[s, j]
            sum_mat[i, j + 1] = total  # 后面的列为每个测试样本Gauss加权之和
    return sum_mat

def output_layer(sum_mat):
    m, n = np.shape(sum_mat)
    output_mat = np.zeros((m, n - 1))
    for i in range(n - 1):
        output_mat[:, i] = sum_mat[:, i + 1] / sum_mat[:, 0]
    return output_mat


def grnn(xlsname, num):
    if num == 0:
        indata, outdata, ninput, noutput = data.load_data(xlsname)
    else:
        indata, outdata, scalerall, ninput, noutput = data.sample_expansion(xlsname, num)
    x_train, x_test, y_train, y_test = train_test_split(indata, outdata, test_size=0.2, random_state=1)
    scaler = preprocessing.StandardScaler()  # 标准化
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    # 模式层
    Eu_dis = distance_mat(x_train, x_test)
    gauss = Gauss(Eu_dis, 0.4)
    # 求和层
    sum_mat = sum_layer(gauss, y_train)
    # 输出层
    output_mat = output_layer(sum_mat)

    R2 = r2_score(y_test, output_mat)
    print('R2:', R2)
    n = len(outdata)  # 样本总数
    p = noutput  # 输出维数
    Adjust_R2 = 1 - (1 - R2) * (n - 1) / (n - p - 1)
    print("Adjust_R2: ", Adjust_R2)
    MAPE = np.mean(np.abs((output_mat - y_test) / y_test))
    print("MAPE: ", MAPE)

    font1 = {'family': 'Times New Roman', 'size': 14}
    r = len(x_test) + 1
    plt.plot(np.arange(1, r), output_mat, 'go-')
    plt.plot(np.arange(1, r), y_test, 'co-')
    plt.title('predict by GRNN', fontdict=font1)
    plt.xlabel('group', fontdict=font1)
    plt.ylabel('damage', fontdict=font1)
    plt.text(8, 0.65, '— predict', fontdict=font1, color='g')
    plt.text(8, 0.55, '— real', fontdict=font1, color='c')
    plt.show()
    ## k-fold
    # kfold = KFold(n_splits=10, random_state=1, shuffle=True)
    # ad_r2 = []
    # mape = []
    # for train, test in kfold.split(indata, outdata):
    #     # 模式层
    #     Eu_dis = distance_mat(indata[train], indata[test])
    #     gauss = Gauss(Eu_dis, 0.4)
    #     # 求和层
    #     sum_mat = sum_layer(gauss, outdata[train])
    #     # 输出层
    #     output = output_layer(sum_mat)
    #
    #     R2 = r2_score(outdata[test], output)
    #     n = len(outdata)  # 样本总数
    #     p = noutput  # 输出维数
    #     Adjust_R2 = 1 - (1 - R2) * (n - 1) / (n - p - 1)
    #     ad_r2.append(Adjust_R2)
    #     MAPE = np.mean(np.abs((output - outdata[test]) / outdata[test]))
    #     mape.append(MAPE)
    # print(" ANN-Adjust_R2: ", np.mean(ad_r2))
    # print(" ANN-MAPE: ", np.mean(mape))


grnn(r'D:\task\Tang\single data(M26ET).xls', 500)

