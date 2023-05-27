# -*- coding: UTF-8 -*-
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os
from sklearn.decomposition import PCA
import data

def create_model(ninput, noutput):
    model = Sequential()
    model.add(Dense(8, activation='relu', input_shape=(ninput,), name='Dense1'))
    model.add(Dense(32, activation='relu', name='Dense2'))
    model.add(Dropout(0.02, name='Dropout1'))
    model.add(Dense(32, activation='relu', name='Dense3'))
    model.add(Dense(16, activation='relu', name='Dense4'))
    model.add(Dense(16, activation='relu', name='Dense5'))
    model.add(Dense(noutput, name='Output'))
    return model


def ann(xlsname, num):
    if num == 0:
        X_data, Y_data, ninput,noutput = data.load_data(xlsname)
    else:
        X_data, Y_data, scaler, ninput, noutput = data.sample_expansion(xlsname, num)
    # pca = PCA(n_components=0.99)
    # X_data = pca.fit_transform(X_data)  # 降维，用处不大
    # 训练ANN模型#
    print('SVR-ANN BEGIN')
    kfold = KFold(n_splits=5, random_state=1, shuffle=True)  # KFold回归，StratifiedKFold分类
    ad_r2 = []
    mape = []
    font1 = {'family': 'Times New Roman', 'size': 14}
    for train, test in kfold.split(X_data, Y_data):
        model = create_model(ninput, noutput)
        model.compile(loss='mse', optimizer='Adam', metrics=['mean_absolute_error'])
        callback_list = [EarlyStopping(monitor='mean_absolute_error', patience=20, verbose=2, mode='min'),
                         ModelCheckpoint(os.path.join(os.path.dirname(xlsname), 'model.hdf5'), monitor='mean_absolute_error',
                         verbose=0, save_best_only=True, mode='min')]
        model.fit(X_data[train], Y_data[train], batch_size=32, epochs=500, verbose=0, callbacks=callback_list)

        Y_pred = model.predict(X_data[test])
        ANNR2 = r2_score(Y_data[test], Y_pred)
        n = len(Y_data)
        p = noutput
        ANNAdjust_R2 = 1 - (1 - ANNR2) * (n - 1) / (n - p - 1)
        ad_r2.append(ANNAdjust_R2)
        ANNMAPE = np.mean(np.abs((Y_pred - Y_data[test]) / Y_data[test]))
        mape.append(ANNMAPE)
        print(" ANN-Adjust_R2: ", np.mean(ad_r2))
        print(" ANN-MAPE: ", np.mean(mape))

#这一步份是在画图，但是建议拿出循环画
    r = len(X_data[test]) + 1
    plt.plot(np.arange(1, r), Y_pred, 'go-')
    plt.plot(np.arange(1, r), Y_data[test], 'co-')
    plt.title('predict by ANN', fontdict=font1)
    plt.xlabel('group', fontdict=font1)
    plt.ylabel('damage', fontdict=font1)
    plt.text(1, 0.65, '— predict', fontdict=font1, color='g')
    plt.text(1, 0.55, '— real', fontdict=font1, color='c')
    plt.show()
    # print(" ANN-Adjust_R2: ", np.mean(ad_r2))
    # print(" ANN-MAPE: ", np.mean(mape))
#    return scaler


ann(r'D:\task\Tang\MFR.xls', 0)
# scaler = ann(r'D:\ABAQUStemp\traindata.xlsx')  # 初始样本再多些，训练准确率会更高
# # 测试
# indata, outdata, ninput, noutput = data.load_data(r'D:\ABAQUStemp\testdata.xlsx')
# model2 = create_model(ninput, noutput)
# model2.load_weights(r'D:\ABAQUStemp\model.hdf5')
# test = scaler.transform(indata)  # 测试集和训练集同一标准
# pred = model2.predict(test)
# MAPE = np.mean(np.abs((pred - outdata) / outdata))
# print('\ntest MAPE', MAPE)

