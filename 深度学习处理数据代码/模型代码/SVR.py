# -*- coding: UTF-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import r2_score
# from sklearn.externals import joblib
import joblib
from sklearn.multioutput import MultiOutputRegressor
from sklearn import preprocessing
import xlrd
import matplotlib.pyplot as plt
import Draw


def svr(xls_path):
    print('SVR BEGIN')
    # 数据集#
    ninput = 0
    noutput = 0
    book = xlrd.open_workbook(xls_path)
    sheet = book.sheet_by_index(0)
    ST = sheet.nrows - 1
    ncol = sheet.ncols - 1
    for i in sheet.row_values(1):
        if i != '':
            ninput += 1
        else:
            noutput = ncol - ninput
            break
    indata = np.zeros((ST, ninput), dtype=np.float_)
    outdata = np.zeros((ST, noutput), dtype=np.float_)
    for j in range(ST):
        data = sheet.row_values(j + 1)
        indata[j] = data[0:ninput]
        outdata[j] = data[ninput + 1:ncol + 1]

    # 训练#
    x_train, x_test, y_train, y_test = train_test_split(indata, outdata, test_size=0.2)
    #    scaler = preprocessing.MinMaxScaler()  # 归一化
    scaler = preprocessing.StandardScaler()  # 标准化
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)  # 测试集和训练集同一标准

    sv = GridSearchCV(SVR(), param_grid={"kernel": ('sigmoid', 'rbf'), "C": np.linspace(0.01, 100, 30),
                                         "gamma": np.linspace(0.0001, 1, 30)}, cv=10)  # start,stop,num. 交叉验证参数寻优

    wrapper = MultiOutputRegressor(sv)  # 简单SVR不支持多输出
    wrapper.fit(x_train, y_train)
    print(wrapper.estimators_[0].best_params_)
    # joblib.dump(wrapper, 'svr.pkl')

    # 预测评估#
    y_pred = wrapper.predict(x_test)
    R2 = r2_score(y_test, y_pred)
    print('R2:', R2)
    n = len(outdata)
    p = noutput  # Adjust_R2消除样本数量影响,R2为负说明模型效果不如取平均
    Adjust_R2 = 1 - (1 - R2) * (n - 1) / (n - p - 1)
    print("Adjust_R2: ", Adjust_R2)
    MAPE = np.mean(np.abs((y_pred - y_test) / y_test))
    print("MAPE: ", MAPE)

    #画折线图#
    font1 = {'family': 'Times New Roman', 'size': 14}
    r = len(x_test) + 1
    plt.plot(np.arange(1, r), y_pred, 'go-')
    plt.plot(np.arange(1, r), y_test, 'co-')
    plt.title('predict by SVR', fontdict=font1)
    plt.xlabel('group', fontdict=font1)
    plt.ylabel('damage', fontdict=font1)
    plt.text(8, 0.65, '— predict', fontdict=font1, color='g')
    plt.text(8, 0.55, '— real', fontdict=font1, color='c')
    plt.show()
    # MCS抽样#
    # Draw.mcs(indata[0],ninput,noutput,scaler,wrapper,sheet.row_values(0)[ninput + 1:ninput + 3])


svr(r'D:\task\Tang\single data(M26ET).xls')
