# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 21:21:02 2020

@author: Jiang Jilian
"""

from utils.normlization import *
# from plot_func import *
from train import *
#import numba
#@numba.jit

if __name__ == '__main__':
    # Load data
    # fileName = "Dataset Final 1026"
    zircons_data = pd.read_excel(dataPath + fileName + ".xlsx")
    zircons_data.loc[zircons_data["zircon"] == "S-type zircon", "label"] = 1
    zircons_data.loc[zircons_data["zircon"] == "I-type zircon", "label"] = 0
    cols = [x for x in zircons_data.index for i in elements if zircons_data.loc[x, i] == 0]
    zircons_data.drop(cols, inplace=True)
    zircons_data.reset_index(inplace=True, drop=True)
    ree_list = ["Y", "Ce", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"]
    zircons_data["REE+Y"] = zircons_data[ree_list].sum(axis=1)
    zircons_data.dropna(subset=elements, inplace=True)
    zircons_data.reset_index(inplace=True, drop=True)
    raw_prediction_set = zircons_data[zircons_data["Set"] == "Prediction set"]
    raw_prediction_set.reset_index(inplace=True, drop=True)

    # Normalize all zircons
    x_train = zircons_data.loc[zircons_data["Set"] == "Training set", elements]
    x_data = preprocess_data(x_train, zircons_data[elements])
    x_data_df = pd.DataFrame(x_data, columns=elements)

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

    # Train model and predict JH zircons and Tibet zircons
    estimators = [RF, SVM, WSVM, TSVM_model]
    estimators_txt = ["RF", "SVM", "WSVM", "TSVM"]
    transform = "_CLR"

    # Set model parameters
    parameters = {"TSVM": {"C": 2.0, "Cl": 3.0},
                  "SVM": {"C": 0.6, "WEIGHT": 1.0},
                  "WSVM": {"C": 1.0, "WEIGHT": 1.0},
                  "RF": {"TREE_NUM": 5, "MAX_LEAF_NODE": 5}}
    # Train models including TSVM、SVM、WSVM、RF,
    # and predict zircons type of testing set and predicting set
    train(estimators, estimators_txt, parameters, zircons_data, train_set, test_set, raw_prediction_set, predict_set)

    # Plot fig1, fig2 and fig 3 in this paper








