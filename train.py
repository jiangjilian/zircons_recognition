# import numpy as np
# import pandas as pd
import os
from sklearn.utils import shuffle
from globalVar import *
from models import *


# 设置文件路径

def train(estimators, estimators_txt, parameters, train_set, test_set, raw_prediction_set, predict_set):
    acc = pd.DataFrame(columns=["method", "parameters", "train_acc", "test_acc"])
    acc["method"] = estimators_txt

    X_train = train_set[elements]
    y_train = train_set['label']
    X_test = test_set[elements]
    y_test = test_set['label']

    X_train, y_train, train_set = shuffle(X_train, y_train, train_set)

    # sample_weight_input2 = np.array(train_set["sample_weight2"])

    for i in np.arange(len(estimators)):
        print(estimators_txt[i])
        # Record Parameters into acc
        parameters_str = ""
        for key in parameters[estimators_txt[i]]:
            parameters_str = parameters_str + str(key) + "=" + str(parameters[estimators_txt[i]][key]) + " "
        acc.loc[i, "parameters"] = parameters_str

        estimator = estimators[i]
        model_parameter = parameters[estimators_txt[i]]
        acc.loc[i, "train_acc"], acc.loc[i, "test_acc"], predict_data, test_data = estimator(X_train, X_test,
                                                                                             y_train, y_test,
                                                                                             predict_set, test_set,
                                                                                             train_set,
                                                                                             **model_parameter)
        new_predict_data = pd.concat([raw_prediction_set[info_list + elements], predict_data["pred_type"]], axis=1)
        if not os.path.exists(outputPath + "test\\"):
            os.makedirs(outputPath + "test\\")
        new_predict_data.to_csv(outputPath + "test\\" + str(estimators_txt[i]) + "_predictData_predict_result.csv")
        test_data.to_csv(outputPath + "test\\" + str(estimators_txt[i]) + "_testData_predict_result.csv")
    acc.to_csv(outputPath + "four_methods_acc.csv")
    print(acc)
# def evaluate():
