import math
import numpy as np
from scipy.stats import stats
from sklearn import preprocessing


def row_ln(row):
    #
    gmean = stats.gmean(np.array(row))
    func = lambda x: math.log(x / gmean, math.e)
    new_row = row.apply(func)
    return new_row


def CLR(x):
    percent_x = (x.T / x.sum(axis=1)).T
    nomalized_x = percent_x.apply(row_ln, axis=1)
    return nomalized_x


def preprocess_data(x_train, x):
    # 对数据进行转换，标准化处理
    # Transform the dataset into standard uniform
    pro_x = CLR(x)
    pro_x_train = CLR(x_train)
    scaler = preprocessing.StandardScaler().fit(pro_x_train)
    X = scaler.transform(pro_x)
    return X
