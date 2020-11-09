
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 10:53:26 2019

@author: Jiang
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import pandas as pd

def gridsearch(data, ):
    #实验次数计数
    """
    test_count = str(4)+"limited_P_20_without_P"

    Path = "F:\\地质所F盘所有资料\\02研二\\2019年9月科研\\锆石项目\\实验\\"
    output = "F:\\地质所F盘所有资料\\02研二\\2019年9月科研\\锆石项目\\实验结果对比\\"
    dataPath = "F:\\地质所F盘所有资料\\02研二\\2019年6月科研\\锆石项目\\data\\"
    dataPath2 = "F:\\地质所F盘所有资料\\02研二\\2019年9月科研\\"
    figPath = "F:\\地质所F盘所有资料\\fig\\comparision\\svm_C\\LOOCV\\"

    #读入数据，并转换成需要的格式
    train_data = pd.read_excel(dataPath2 + "new_train_data.xlsx")
    data = train_data

    pred_data = pd.read_excel(dataPath + "new_predict1.xlsx")
    #pred_data = pd.read_excel(dataPath + "new_predict1.xlsx")

    elements1 = ["P","Y" , "Ce",  "Nd", "Sm", "Eu", "Gd","Tb", "Dy","Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Th", "U", "Ti"]
    elements2 = ["P","Y" , "Ce",  "Nd", "Sm", "Eu", "Gd","Tb", "Dy","Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Th", "U"]
    elements3 = ["Y" ,  "Sm", "Eu", "Gd","Tb", "Dy","Ho", "Er", "Tm", "Ti", "Th", "U","Lu/Hf","Ce/Nd","Yb/Pr"]
    elements4 = ["Y" , "Ce",  "Nd", "Sm", "Eu", "Gd","Tb", "Dy","Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Th", "U", "Ti"]
    elements5 = ["Y" , "Ce",  "Nd", "Sm", "Eu", "Gd","Tb", "Dy","Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Th", "U", "Ti","Th/U", "Lu/Hf","Ce/Nd",]
    elements6 = ["P","Y" ,"Nd", "Sm", "Lu", "Hf"]
    #removed_elements = ["P", "Ce",  "Nd", "Sm", "Eu", "Hf", "Th", "U"]
    elements = elements2
    """
    X = data[elements]
    y = data['label']
    # 将数据集分成训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    if estimator == SVM:

        # 设置gridsearch的参数
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [10, 1, 1e-1, 1e-2, 1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
                            {'kernel': ['linear'], 'C': [1, 2, 3, 4, 5,6, 7, 8, 9,10, 20,40,80, 100,200,400,500, 600, 700,800, 1000]}]

        #设置模型评估的方法.如果不清楚,可以参考上面的k-fold章节里面的超链接
        f = open("./parameters_info.txt", 'w+')
        #print(tuned_parameters, file = f)
        scores = ['precision', 'recall']

        for score in scores:
            print("# Tuning hyper-parameters for %s" % score, file = f)

            #构造这个GridSearch的分类器,5-fold
            clf = GridSearchCV(SVC(), tuned_parameters, cv=3, scoring='%s_weighted' % score)
            #只在训练集上面做k-fold,然后返回最优的模型参数
            clf.fit(X_train, y_train)

            print("Best parameters set found on development set:", file = f)
            #输出最优的模型参数
            print(clf.best_params_, file = f)
            print("Grid scores on development set:", file = f)
            for params, mean_score, scores in zip(clf.cv_results_['params'],clf.cv_results_['mean_train_score'],clf.cv_results_['mean_test_score']):
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean_score, scores.std() * 2, params), file = f)

            print("Detailed classification report:", file = f)
            print("The model is trained on the full development set.", file = f)
            print("The scores are computed on the full evaluation set.", file = f)
            #在测试集上测试最优的模型的泛化能力.
            y_true, y_pred = y_test, clf.predict(X_test)
            print(classification_report(y_true, y_pred), file = f)
    else:


def get_parameters():

    return parameters
