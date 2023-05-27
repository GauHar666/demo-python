
  # -*- coding: UTF-8 -*-
import numpy as np
from sklearn.model_selection import GridSearchCV#GridSearchCV用于系统地遍历模型的多种参数组合，通过交叉验证确定最佳参数
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn import preprocessing
import xlrd#导入读取excel表格的模块
import random
from scipy.stats import norm


def load_data(xls_path):
    ninput = 0
    noutput = 0
    book = xlrd.open_workbook(xls_path) ##打开excel表，返回data对象，括号内部填excel表的路径
    sheet = book.sheet_by_index(0)  ##通过索引顺序获取   sheet_by_index(0)为第一个sheet  同理sheet_by_index(1)为第二个sheet 获取的sheet返回的是16进制地址
    ST = sheet.nrows - 1  #返回表格的行数-1
    ncol = sheet.ncols - 1  #返回列数-1
    for i in sheet.row_values(1):  ##获取指定行，返回列表，这里是第一行中每个值做循环
        if i != '':#用空格分割
            ninput += 1
        else:
            noutput = ncol - ninput
            break
    indata = np.zeros((ST, ninput), dtype=np.float_)
    outdata = np.zeros((ST, noutput), dtype=np.float_)#创建X,Y张量，初始化为0
    for j in range(ST):#每一行做循环
        data = sheet.row_values(j + 1)#获取指定行，返回列表，每行都返回一个列表
        indata[j] = data[0:ninput]#把对应行填入X输入值
        outdata[j] = data[ninput + 1:ncol + 1]#对应列填入输出值
    return indata, outdata, ninput, noutput#返回X,Y张量和影响变量数X和Y的数目


def sample_expansion(xls_path, n_sample):
    indata, outdata, ninput, noutput = load_data(xls_path)  # all data used to train
    scaler = preprocessing.StandardScaler()  # 标准化
    x_train = scaler.fit_transform(indata) #找出indata的均值和标准差，并应用在X_train上
    # 训练SVR#
    sv = GridSearchCV(SVR(kernel='rbf'), param_grid={"C": np.linspace(0.01, 200, 30),
                "gamma": np.linspace(0.00001, 1, 30)}, cv=5)  # start,stop,num. 网络搜索+交叉验证参数寻优
    wrapper = MultiOutputRegressor(sv)  # 简单SVR不支持多输出
    wrapper.fit(x_train, outdata)
    print(wrapper.estimators_[0].best_params_)

    # MCS抽样#
    X = indata[0]  # 数据第一行为均值
    X11 = np.zeros((n_sample, ninput), dtype=np.float_)
    Y = np.zeros((n_sample, noutput), dtype=np.float_)
    for i in range(n_sample):
        for j in range(ninput):
            X11[i, j] = norm.ppf((random.random()), loc=X[j], scale=0.03 * X[j])
    X11 = scaler.transform(X11)  # 标准化

    for j in range(n_sample):
        Y[j] = wrapper.predict(X11[j].reshape(1, -1))

    X_data = np.concatenate((X11, scaler.transform(indata)), axis=0)
    Y_data = np.concatenate((Y, outdata), axis=0)
    return X_data, Y_data, scaler, ninput, noutput


#load_data(r'D:\task\Tang\MFR.xls')



