import joblib
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from globalVar import *


class TSVM(SVC):
    def __init__(self, C, Cl, kernel="linear"):
        """
        Initial TSVM
        Parameters
        ----------
        kernel: kernel of svm
        Cl Cu and C: the hyper-parameters
        """
        self.Cl, self.Cu = Cl, 0.001
        self.kernel = kernel
        self.clf = svm.SVC(C=C, kernel=self.kernel)

    def load(self, model_path='./TSVM.model'):
        """
        Load TSVM from model_path
        Parameters
        ----------
        model_path: model path of TSVM
                        model should be svm in sklearn and saved by sklearn.externals.joblib
        """
        self.clf = joblib.load(model_path)

    def train(self, X1, Y1, X2):
        """
        Train TSVM by X1, Y1, X2
        Parameters
        ----------
        X1: Input data with labels
                np.array, shape:[n1, m], n1: numbers of samples with labels, m: numbers of features
        Y1: labels of X1
                np.array, shape:[n1, ], n1: numbers of samples with labels
        X2: Input data without labels
                np.array, shape:[n2, m], n2: numbers of samples without labels, m: numbers of features
        """
        N = len(X1) + len(X2)
        sample_weight = np.ones(N)
        sample_weight[len(X1):] = self.Cu

        self.clf.fit(X1, Y1)
        Y2 = self.clf.predict(X2)
        Y1 = np.expand_dims(Y1, 1)
        Y2 = np.expand_dims(Y2, 1)
        X2_id = np.arange(len(X2))
        X3 = np.vstack([X1, X2])
        Y3 = np.vstack([Y1, Y2])

        while self.Cu < self.Cl:
            self.clf.fit(X3, Y3, sample_weight=sample_weight)
            while True:
                Y2_d = self.clf.decision_function(X2)  # linear: w^Tx + b
                Y2 = Y2.reshape(-1)
                epsilon = 1 - Y2 * Y2_d  # calculate function margin
                positive_set, positive_id = epsilon[Y2 > 0], X2_id[Y2 > 0]
                negative_set, negative_id = epsilon[Y2 < 0], X2_id[Y2 < 0]
                positive_max_id = positive_id[np.argmax(positive_set)]
                negative_max_id = negative_id[np.argmax(negative_set)]
                a, b = epsilon[positive_max_id], epsilon[negative_max_id]
                if a > 0 and b > 0 and a + b > 2.0:
                    Y2[positive_max_id] = Y2[positive_max_id] * -1
                    Y2[negative_max_id] = Y2[negative_max_id] * -1
                    Y2 = np.expand_dims(Y2, 1)
                    Y3 = np.vstack([Y1, Y2])
                    self.clf.fit(X3, Y3, sample_weight=sample_weight)
                else:
                    break
            self.Cu = min(2 * self.Cu, self.Cl)
            sample_weight[len(X1):] = self.Cu

    def score(self, X, Y):
        """
        Calculate accuracy of TSVM by X, Y
        Parameters
        ----------
        X: Input data
                np.array, shape:[n, m], n: numbers of samples, m: numbers of features
        Y: labels of X
                np.array, shape:[n, ], n: numbers of samples
        Returns
        -------
        Accuracy of TSVM
                float
        """
        return self.clf.score(X, Y)

    def predict(self, X):
        """
        Feed X and predict Y by TSVM
        """
        return self.clf.predict(X)

    def predict_proba(self, X):
        """
        Feed X and predict Y by TSVM            
        """

    def save(self, path='./TSVM.model'):
        """
        Save TSVM to model_path
        """
        joblib.dump(self.clf, path)

    def decision_function(self, X):
        """
        Feed X and predict Y by TSVM
        """
        return self.clf.decision_function(X)

    def classification_report(self, X):
        return self.clf.classification_report(X)


def get_weight_list(train_set, WEIGHT):
    weighted_spot = train_set[(train_set["P_copy"] <= 20) & (train_set["REE+Y"] <= 20)].index

    sample_weight = np.ones(len(train_set))

    sample_weight[weighted_spot] = np.ones(len(weighted_spot)) * WEIGHT

    sample_weight_list = np.array(sample_weight)

    return sample_weight_list


def SVM(x_train, x_test, y_train, y_test, pred_data, test_data, train_data, **kw):
    """
    Feed X and predict Y by TSVM
    Parameters
    ----------
    X: Input data
            np.array, shape:[n, m], n: numbers of samples, m: numbers of features
    Returns
    -------
    train_acc, test_acc, pred_data, test_data
            np.array, shape:[n, ], n: numbers of samples
            :param elements:
    """
    # x_train = preprocess_data(x_train)
    # x_test = preprocess_data(x_test)
    pred_x = pred_data[elements]
    sample_weight_list = get_weight_list(train_data, WEIGHT=kw['WEIGHT'])
    # pred_x = preprocess_data(pred_x)

    # C = 0.6
    mykernel = "linear"

    svm_model = svm.SVC(C=kw['C'], kernel=mykernel, probability=True, decision_function_shape='ovr')
    svm_model.fit(x_train, y_train.ravel(), sample_weight=sample_weight_list)

    train_acc = accuracy_score(y_train, svm_model.predict(x_train))
    test_acc = accuracy_score(y_test, svm_model.predict(x_test))

    y_test_pred = svm_model.predict(x_test)
    y_predict = svm_model.predict(pred_x)
    y_train_pred = svm_model.predict(x_train)
    pred_data["pred_type"] = y_predict
    test_data["pred_type"] = y_test_pred
    train_data["pred_type"] = y_train_pred
    # print('Accuracy:', accuracy_score(y_test, svm_model.predict(x_test)))
    return train_acc, test_acc, pred_data, test_data


def WSVM(x_train, x_test, y_train, y_test, pred_data, test_data, train_data, **kw):
    train_acc, test_acc, pred_data, test_data = SVM(x_train, x_test, y_train, y_test, pred_data, test_data, train_data,
                                                    **kw)
    return train_acc, test_acc, pred_data, test_data


def TSVM_model(x_train, x_test, y_train, y_test, pred_data, test_data, train_data, **kw):
    pred_x = pred_data[elements]
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    y_train[y_train == 0] = -1
    y_test[y_test == 0] = -1

    model = TSVM(C=kw["C"], Cl=kw["Cl"])
    #model.initial()
    # pred_x = pred_data.loc[pred_data["AGE"]<4000,elements]
    pred_test_x = pd.concat([pred_x, x_test], axis=0)
    pred_test_x.reset_index(inplace=True, drop=True)
    model.train(x_train, y_train, pred_test_x)

    train_acc = model.score(x_train, y_train)
    # print("train accuracy:%f" %train_accuracy)
    test_acc = model.score(x_test, y_test)

    y_test_pred = model.predict(x_test)
    y_train_pred = model.predict(x_train)
    y_predict = model.predict(pred_x)

    y_test_pred[y_test_pred == -1] = 0
    y_train_pred[y_train_pred == -1] = 0
    y_predict[y_predict == -1] = 0
    #print(y_predict)
    pred_data["pred_type"] = y_predict
    test_data["pred_type"] = y_test_pred
    train_data["pred_type"] = y_train_pred

    # print("test accuracy:%f" %test_accuracy)

    # print('Accuracy:', accuracy_score(y_test, svm_model.predict(x_test)))
    return train_acc, test_acc, pred_data, test_data


def RF(x_train, x_test, y_train, y_test, pred_data, test_data, train_data, **kw):
    # x_train = preprocess_data(x_train)
    # x_test = preprocess_data(x_test)
    pred_x = pred_data[elements]
    ##pred_x = preprocess_data(pred_x)
    # print(kw)
    TREE_NUM = int(kw["TREE_NUM"])
    # MAX_LEAF_NODE = kw["MAX_LEAF_NODE"]
    clf = RandomForestClassifier(n_estimators=TREE_NUM,
                                 oob_score=True,
                                 # max_leaf_nodes=MAX_LEAF_NODE,
                                 random_state=789)

    # print(x_train)
    # print(y_train)
    clf.fit(x_train, y_train)

    # 对训练集和验证集进行预测
    train_score = clf.score(x_train, y_train)
    test_score = clf.score(x_test, y_test)

    y_test_pred = clf.predict(x_test)
    y_predict = clf.predict(pred_x)
    y_train_pred = clf.predict(x_train)
    # y_test_pred[y_test_pred==-1]=0
    # y_predict[y_predict==-1]=0
    # y_train_pred[y_train_pred==-1]=0

    pred_data["pred_type"] = y_predict
    test_data["pred_type"] = y_test_pred
    train_data["pred_type"] = y_train_pred
    return train_score, test_score, pred_data, test_data
