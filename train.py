from sklearn.utils import shuffle
from models import *
from globalVar import *


def train(estimators, estimators_txt, parameters, raw_data, train_set, test_set, raw_prediction_set, predict_set):
    acc = pd.DataFrame(columns=["method", "parameters", "train_acc", "test_acc"])
    acc["method"] = estimators_txt

    X_train = train_set[elements]
    y_train = train_set['label']
    X_test = test_set[elements]
    y_test = test_set['label']

    X_train, y_train, train_set = shuffle(X_train, y_train, train_set)

    # sample_weight_input2 = np.array(train_set["sample_weight2"])
    all_prediction_data = raw_data
    for i in np.arange(len(estimators)):
        print(estimators_txt[i])
        # Record Parameters into acc
        parameters_str = ""
        for key in parameters[estimators_txt[i]]:
            parameters_str = parameters_str + str(key) + "=" + str(parameters[estimators_txt[i]][key]) + " "
        acc.loc[i, "parameters"] = parameters_str

        estimator = estimators[i]
        model_parameter = parameters[estimators_txt[i]]
        acc.loc[i, "train_acc"], acc.loc[i, "test_acc"], predict_data, test_data_pred = estimator(X_train, X_test,
                                                                                                  y_train, y_test,
                                                                                                  predict_set, test_set,
                                                                                                  train_set,
                                                                                                  **model_parameter)
        predict_data_pred = pd.concat([raw_prediction_set[info_list + elements], predict_data["pred_type"]], axis=1)
        predict_data_pred.to_csv(outputPath + str(estimators_txt[i]) + "_predictData_predict_result.csv")
        test_data_pred.to_csv(outputPath + str(estimators_txt[i]) + "_testData_predict_result.csv")
        all_prediction = pd.concat([predict_data_pred, test_data_pred], ignore_index=True, axis=0)
        all_prediction_data = all_prediction_data.merge(all_prediction[["No", "pred_type"]], on=["No"], how="outer",
                                       suffixes=("", "_" + estimators_txt[i]))

        JH_pred_type = predict_data_pred.loc[predict_data_pred["Rock type"] == "JH zircon", "pred_type"]
        JH_S_ratio = JH_pred_type.value_counts()[1] / (JH_pred_type.value_counts()[0] + JH_pred_type.value_counts()[1])
        acc.loc[i, "JH_S_ratio"] = JH_S_ratio
        # Tb_pred_type = predict_data_pred.loc[predict_data_pred["Rock type"] == "detrital zircon (<150Ma)", "pred_type"]
        # Tb_S_ratio = Tb_pred_type.value_counts()[1] / (Tb_pred_type.value_counts()[0] + Tb_pred_type.value_counts()[1])
        # acc.loc[i, "Tb_S_ratio"] = Tb_S_ratio

    acc.to_csv(outputPath + "four_methods_acc.csv")
    all_prediction_data.to_csv(dataPath + fileName + "_prediction.csv", index=False)
    print(acc)
    print(all_prediction_data.columns)
# def evaluate():
