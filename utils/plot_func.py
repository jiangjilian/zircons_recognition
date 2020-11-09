import os

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from globalVar import *
#import seaborn as sns
#import matplotlib.patches as mpatches


# def plot_learning_curve():
def ML_JH_zircon_plot():
    df_ml_jh_plot = pd.read_excel("Dataset Final1101.xlsx", sheet_name='Sheet 1')
    df_ml_jh_plot.columns
    ml_ziron_type = [
        'S-type zircon',
        'I-type zircon'
    ]

    colors_JH = [
        'steelblue',
        'red'
    ]
    loc = plticker.MultipleLocator(base=200.0)  # this locator puts ticks at regular intervals

    bins = range(3300, 4400, 50)

    x1 = list(df_ml_jh_plot[df_ml_jh_plot['Machine learning type '] == 'S-type zircon']['Age（Ma)'])
    x2 = list(df_ml_jh_plot[df_ml_jh_plot['Machine learning type '] == 'I-type zircon']['Age（Ma)'])
    f, ax = plt.subplots()
    plt.hist(
        [x1, x2],
        bins=bins,
        stacked=True,
        #     normed=True,
        color=colors_JH,
        alpha=0.5,
        edgecolor="w",
        linewidth=1,
        label=ml_ziron_type
    )

    plt.legend(loc='upper left')
    plt.xlabel('Age(Ma)', fontsize=13)
    plt.ylabel('Frequency', fontsize=13)
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(plticker.MultipleLocator(base=5))
    plt.xlim(3300, 4400)
    # plt.ylim(0, 20)
    ax.tick_params(axis="y", direction="in")
    ax.tick_params(axis="x", direction="in")

    plt.savefig('ML_JH_stacked_hist.pdf', transparent=True)
    plt.show()


def ML_Qinghai_Tibet_Plateau_Zircon_PLOT():
    df_ml_qtp_plot = pd.read_excel('Dataset Final1101.xlsx', sheet_name='Sheet 2')
    bins = range(0, 150, 10)

    x1 = list(df_ml_qtp_plot[df_ml_qtp_plot['Machine learning type '] == 'S-type zircon']['Age（Ma)'])
    x2 = list(df_ml_qtp_plot[df_ml_qtp_plot['Machine learning type '] == 'I-type zircon']['Age（Ma)'])
    f, ax = plt.subplots()
    plt.hist(
        [x1, x2],
        bins=bins,
        stacked=True,
        #     normed=True,
        color=colors_JH,
        alpha=0.5,
        edgecolor="w",
        linewidth=1,
        label=ml_ziron_type
    )

    plt.legend(loc='upper left')
    plt.xlabel('Age(Ma)', fontsize=13)
    plt.ylabel('Frequency', fontsize=13)
    # ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(plticker.MultipleLocator(base=30))
    plt.xlim(0, 150)
    # plt.ylim(0, 20)
    ax.tick_params(axis="y", direction="in")
    ax.tick_params(axis="x", direction="in")

    plt.savefig('ML_QTP_stacked_hist.pdf', transparent=True)
    plt.show()


def JH_element_vs_element_plot(element1, element2):
    JH_zircons = pd.read_excel('Dataset Final1101.xlsx', sheet_name='Sheet 3')
    type_list = [
        'I*',
        'I-type zircon',
        'S*',
        'S-type zircon',
    ]
    colors = [
        'red',
        'red',
        'steelblue',
        'steelblue',
    ]
    facecolors = [
        'none',
        'red',
        'none',
        'steelblue',
    ]
    for t, c1, c2 in zip(type_list, colors, facecolors):
        plt.scatter(
            JH_zircons[JH_zircons["zircon "] == t]["Hf (mol%)"],
            JH_zircons[JH_zircons["zircon "] == t]["P (mol%)"],
            s=JH_zircons[JH_zircons["zircon "] == t]["P (μmol/g)"],
            facecolors=c2,
            edgecolors=c1,
            #         alpha=0.8,
            label=t
        )
    plt.legend()
    for handle in lgnd.legendHandles:
        handle.set_sizes([48.0])

    plt.xlim(20, 100)
    plt.ylim(0, 40)
    plt.xlabel('Hf (mol%)', fontsize=13)
    plt.ylabel('P (mol%)', fontsize=13)
    plt.savefig('JH_P vs Hf.eps')


def plot_predict_one(figpath, fig, title, pred_, filename, test_count, raw_num, col_num, count, max_x, max_y):
    # fig2 = plt.figure(figsize=(16,12))
    # ax = fig2.add_subplot(111)
    ax = plt.subplot(raw_num, col_num, count)
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_linewidth(3)
    ax.spines['right'].set_linewidth(3)
    ax.spines['top'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)
    ax.tick_params(direction='out', length=15, width=2, colors='black', grid_color='b', pad=10)
    # cm = plt.cm.get_cmap('RdYlBu')
    # I型绿色，S型红色 {"I_type":'#3CC9CF',"S_type":'#F2766E'}
    plt.scatter(pred_.loc[pred_["pred_type"] == 0, "P_copy"], pred_.loc[pred_["pred_type"] == 0, "REE+Y"], c="#3CC9CF",
                marker='o', s=800, alpha=1, label="Predicted I-type")
    plt.scatter(pred_.loc[pred_["pred_type"] == 1, "P_copy"], pred_.loc[pred_["pred_type"] == 1, "REE+Y"], c="#F2766E",
                marker='o', s=800, alpha=1, label="Predicted S-type")
    # 深蓝色#4A708B
    # 暗红色#8B2323

    # 绘制4条直线
    x = np.arange(0.0, max_x, 0.01)
    f1 = x - 7.33  # S下界限
    f2 = x + 5.64  # S上界限
    # f3 = 3.10 * x #I的上界限
    f4 = x
    # 红色
    plt.plot(x, f1, color="#B22222", linewidth=2, linestyle="--")
    # 绿色
    plt.plot(x, f2, color="#2E8B57", linewidth=2, linestyle="--")
    # 蓝色
    # plt.plot(x, f3,color = "#5CACEE",linewidth=2, linestyle="--")
    # 灰色
    plt.plot(x, f4, color="#ADADAD", linewidth=2, linestyle="--")
    plt.xlim(0, max_x)
    plt.ylim(0, max_y)
    plt.xlabel("P(μmol/g)", fontsize=40, labelpad=10, weight='normal')
    plt.ylabel("REE+Y(μmol/g)", fontsize=40, labelpad=10, weight='normal')
    plt.xticks(fontsize=35, color='black')
    plt.yticks(fontsize=35, color='black')
    plt.legend(loc="best", fontsize=35)
    plt.title(title, fontsize=45)
    plt.tight_layout()
    return plt


def plot_predict(predict_data, test_count, estimators, xlim, ylim):
    fig = plt.figure(figsize=(38, 28))
    raw_num, col_num = 2, 2
    fig.subplots_adjust(wspace=0.3, hspace=0.2)
    if not os.path.exists(figPath):
        os.makedirs(figPath)
    # fig.savefig(figPath + test_count + "_all_predict_four_models_seeds.svg")
    fig.savefig(figPath + test_count + "_all_predict_four_models_seeds.jpg")

    fig2 = plt.figure(figsize=(38, 28))
    for j in np.arange(len(estimators)):
        test_data = pd.read_csv(outputPath + str(col[j]) + test_count + "_testData_predict_result.csv", sep=',',
                                engine='python')
        plt = plot_predict_one(figPath + str(col[j]) + "\\", fig2, col[j], test_data,
                               col[j] + "_test_data_" + "_seed_" + str(2), "compare", raw_num, col_num, j + 1, xlim,
                               ylim)
    fig2.subplots_adjust(wspace=0.3, hspace=0.2)

    # fig2.savefig(figPath + test_count + "SC_HMM_LOOCV" + "_all_predic_four_models_seeds.svg")
    fig2.savefig(figPath + test_count + "SC_HMM_LOOCV" + "_all_predict_four_models_seeds.jpg")
