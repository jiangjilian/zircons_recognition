# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 21:21:02 2020

@author: Jiang Jilian
"""

from utils.normlization import *
#from utils.plot_func import *
from train import *
#import numba
#@numba.jit

if __name__ == '__main__':
    # 读入数据，并转换成需要的格式
    zircons_data = pd.read_excel(dataPath + "Dataset Final 1026.xlsx")
    zircons_data.loc[zircons_data["zircon"] == "S-type zircon", "label"] = 1
    zircons_data.loc[zircons_data["zircon"] == "I-type zircon", "label"] = 0
    # raw_data = raw_data[~raw_data[element].isin(1) for element in elements]
    cols = [x for x in zircons_data.index for i in elements if zircons_data.loc[x, i] == 0]
    zircons_data.drop(cols, inplace=True)
    zircons_data.reset_index(inplace=True, drop=True)
    ree_list = ["Y", "Ce", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"]
    zircons_data["REE+Y"] = zircons_data[ree_list].sum(axis=1)
    zircons_data.dropna(subset=elements, inplace=True)
    zircons_data.reset_index(inplace=True, drop=True)
    raw_prediction_set = zircons_data[zircons_data["Set"] == "Prediction set"]
    raw_prediction_set.reset_index(inplace=True, drop=True)

    # normalize all zircons
    x_train = zircons_data.loc[zircons_data["Set"] == "Training set", elements]
    x_data = preprocess_data(x_train, zircons_data[elements])
    x_data_df = pd.DataFrame(x_data, columns=elements)
    # x_prediction = preprocess_data(x_train, raw_prediction_set[elements])

    # all_data.rename(columns={'P':'p'}, inplace = True)
    zircons_data["P_copy"] = zircons_data["P"].copy()

    data = pd.concat(
        [x_data_df, zircons_data[info_list + ["P_copy"]]],
        axis=1)
    train_set = data[(data["Set"] == "Training set")]
    train_set.reset_index(inplace=True, drop=True)
    test_set = data[(data["Set"] == "Testing set")]
    test_set.reset_index(inplace=True, drop=True)
    predict_set = data[(data["Set"] == "Prediction set")]
    predict_set.reset_index(inplace=True, drop=True)
    print("--------------------------------")

    X = np.array(data[elements])
    y = np.array(data["label"])

    estimators = [RF, SVM, WSVM, TSVM_model]
    estimators_txt = ["RF", "SVM", "WSVM", "TSVM"]
    transform = "_CLR"

    # 设置模型参数
    parameters = {"TSVM": {"C": 1.0, "Cl": 1.0},
                  "SVM": {"C": 1.0, "WEIGHT": 1.0},
                  "WSVM": {"C": 1.0, "WEIGHT": 1.0},
                  "RF": {"TREE_NUM": 1}}
    #训练模型并预测
    train(estimators, estimators_txt, parameters, train_set, test_set, raw_prediction_set, predict_set)






