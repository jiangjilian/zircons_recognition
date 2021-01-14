import os
import math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
# import matplotlib.patches as mpatches
# import seaborn as sns

from globalVar import *


def ree_boxplot():
    element_list = [
        'Hf', 'Y', 'P', 'U', 'Th', 'Lu', 'Yb', 'Tm', 'Er', 'Ho', 'Dy', 'Tb', 'Gd', 'Eu', 'Sm', 'Nd', 'Pr',
        'Ce', 'La', 'Ti', 'Nb', 'Ta', 'Pb', 'Fe', 'Ca', 'Sc', 'Li', 'Sr', 'B', 'Rb'
    ]
    # read zircon data
    df_zircon = pd.read_excel(dataPath + "zircon_dataset.xlsx", sheet_name='Sheet1')
    # read n/N data (Extended Data Table 1)
    df_n_percent = pd.read_excel(dataPath + 'data_count_percent.xlsx')
    bar_color_index = df_n_percent[element_list].iloc[0, :].values
    fig_box_data = np.log10(df_zircon[element_list])
    # exclude the highest and lowest 20% of values of each element to reduce scatter
    low, high = fig_box_data.quantile([.2, .8]).values
    data_plot = []
    for i in range(len(element_list)):
        data_plot.append(
            fig_box_data[(fig_box_data[fig_box_data.columns[i]] < high[i]) & (
                    fig_box_data[fig_box_data.columns[i]] > low[i])].iloc[:, i].values
        )
    df = pd.DataFrame(data_plot).T
    df.columns = element_list
    df_test = pd.DataFrame(data=list(df['Hf'].dropna()), columns=['Data'])
    df_test['ID'] = 'Hf'
    df_test['n%'] = bar_color_index[0]
    for i in range(1, len(element_list)):
        label = element_list[i]
        df_t = pd.DataFrame(data=list(df[label].dropna()), columns=['Data'])
        df_t['ID'] = label
        df_t['n%'] = bar_color_index[i]
        df_test = pd.concat([df_test, df_t])
    # box plot
    cmap_name = 'RdBu'
    rb_cm = cm.get_cmap(cmap_name)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.2)
    ## create a scalarmappable from the colormap
    sm = matplotlib.cm.ScalarMappable(cmap=rb_cm, norm=norm)
    sm.set_array([])
    flierprops = dict(marker='o', markerfacecolor='g', markersize=0.5, linestyle='none')
    medianprops = dict(linestyle='--', linewidth=.5, color='k')
    boxprops = dict(linestyle='-', linewidth=0, color='darkgoldenrod')
    fig, ax = plt.subplots(figsize=(16, 6))
    box1 = plt.boxplot(
        data_plot,
        labels=element_list,
        flierprops=flierprops,
        medianprops=medianprops,
        boxprops=boxprops,
        patch_artist=True
    )
    for patch, i in zip(box1['boxes'], range(len(element_list))):
        patch.set_facecolor(
            rb_cm(norm(bar_color_index[i]))
        )
    plt.xticks(size=12)
    plt.ylabel('Content (' + r'$\mu$' + 'mol/g)', size=15)
    plt.ylim(-4.5, 2.5)
    y = list(range(-4, 3))
    plt.yticks(
        y,
        [r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$', r'$10^{1}$', r'$10^{2}$']
    )
    # plot color bar
    cbaxes = fig.add_axes([0.45, 0.8, 0.2, 0.05])
    cbar = ax.figure.colorbar(
        sm,
        cax=cbaxes,

        ticks=list(np.arange(0, 1.2, 0.2)),
        orientation='horizontal',
    )
    cbar.set_label(label='Trace element availability(%)', size=15)
    if not os.path.exists(figPath):
        os.makedirs(figPath)
    plt.savefig(figPath + "trace_elements_content_boxplot.png")
    plt.show()


def ML_JH_zircon_P_or_Ree_Hf_plot():
    # ## JH_P(REE+Y) vs Hf_plot
    df_JH_plot = pd.read_excel(dataPath + 'Dataset 0104_plot.xlsx', sheet_name='Sheet 2')

    type_list = [
        'I*',
        'I-type zircon',
        'S*',
        'S-type zircon',
        'TTG zircon'
    ]
    colors = [
        'red',
        'red',
        'steelblue',
        'steelblue',
        'g'
    ]
    facecolors = [
        'none',
        'red',
        'none',
        'steelblue',
        'none'
    ]
    f, ax = plt.subplots(figsize=(4, 3))
    for t, c1, c2 in zip(type_list, colors, facecolors):
        plt.scatter(
            df_JH_plot[df_JH_plot["zircon "] == t]["Hf (mol%)"],
            df_JH_plot[df_JH_plot["zircon "] == t]["P (mol%)"],
            s=df_JH_plot[df_JH_plot["zircon "] == t]["P (μmol/g)"],
            facecolors=c2,
            edgecolors=c1,
            linewidth=0.5,
            #         alpha=0.8,
            label=t
        )
    lgnd = plt.legend(fontsize=8)
    for handle in lgnd.legendHandles:
        handle.set_sizes([10])

    plt.xlim(20, 100)
    plt.ylim(0, 40)
    ax.yaxis.set_major_locator(plticker.MultipleLocator(base=10))
    ax.tick_params(axis="y", direction="in")
    ax.tick_params(axis="x", direction="in")
    plt.xlabel('Hf (mol%)', fontsize=13)
    plt.ylabel('P (mol%)', fontsize=13)
    # plt.tight_layout()
    plt.savefig(figPath + 'JH_P vs Hf0104.png')
    plt.show()

    f, ax = plt.subplots(figsize=(4, 3))
    for t, c1, c2 in zip(type_list, colors, facecolors):
        sc = plt.scatter(
            df_JH_plot[df_JH_plot["zircon "] == t]["Hf (mol%)"],
            df_JH_plot[df_JH_plot["zircon "] == t]["(REE+Y)3+ (mol%)"],
            s=df_JH_plot[df_JH_plot["zircon "] == t]["P (μmol/g)"],
            facecolors=c2,
            edgecolors=c1,
            linewidth=0.5,
            #            alpha=0.8,
            label=t
        )
    b1 = plt.scatter([], [], s=15, marker='o', facecolors='none', edgecolors='steelblue', linewidth=0.5)
    b2 = plt.scatter([], [], s=30, marker='o', facecolors='none', edgecolors='steelblue', linewidth=0.5)
    b3 = plt.scatter([], [], s=45, marker='o', facecolors='none', edgecolors='steelblue', linewidth=0.5)
    b4 = plt.scatter([], [], s=60, marker='o', facecolors='none', edgecolors='steelblue', linewidth=0.5)
    plt.legend((b1, b2, b3, b4),
               ('15 μmol/g', '30 μmol/g', '45 μmol/g', '60 μmol/g'),
               scatterpoints=1,
               loc='upper right',
               ncol=1,
               fontsize=8)
    plt.xlim(20, 100)
    plt.ylim(0, 50)
    ax.yaxis.set_major_locator(plticker.MultipleLocator(base=10))
    ax.tick_params(axis="y", direction="in")
    ax.tick_params(axis="x", direction="in")
    plt.xlabel('Hf (mol%)', fontsize=13)
    plt.ylabel('(REE+Y)3+ (mol%)', fontsize=13)
    # plt.tight_layout()
    plt.savefig(figPath + 'JH_REE vs Hf0104.png')
    plt.show()


def ALL_ZIRCONS_P_VS_AGE():
    # ## Age vs P
    df_age_plot = pd.read_excel(dataPath + 'Dataset 0104_plot.xlsx', sheet_name='Sheet 3')
    ziron_type = [
        'S-type zircon',
        'I-type zircon'
    ]

    colors = [
        'steelblue',
        'red'
    ]
    f, ax = plt.subplots(figsize=(4, 3))
    for t, c in zip(ziron_type, colors):
        plt.scatter(
            df_age_plot[df_age_plot["Type"] == t]["Age"] / 1000,
            df_age_plot[df_age_plot["Type"] == t]["P (μmol/g)"],
            s=10,
            facecolors="none",
            edgecolors=c,
            linewidth=0.5,
            #         alpha=0.8,
            label=t,
            clip_on=False,
            zorder=10,
        )
    lgnd = plt.legend()
    for handle in lgnd.legendHandles:
        handle.set_sizes([10.0])
    plt.xlim(0, 4.5)
    plt.ylim(0, 70)
    ax.tick_params(axis="y", direction="in")
    ax.tick_params(axis="x", direction="in")
    ax.xaxis.set_major_locator(plticker.MultipleLocator(base=0.5))
    ax.yaxis.set_major_locator(plticker.MultipleLocator(base=10))
    plt.xlabel('Age (Ga)', fontsize=13)
    plt.ylabel('P (μmol/g)', fontsize=13)
    # plt.tight_layout()
    plt.savefig(figPath + 'Age vs P.png')
    plt.show()


def ML_JH_zircon_hist_plot():
    df_ml_jh_plot = pd.read_excel(dataPath + "Dataset 0104_plot.xlsx", sheet_name='Sheet 1')
    ml_ziron_type = [
        'S-type zircon',
        'I-type zircon'
    ]
    colors_JH = [
        'steelblue',
        'red'
    ]
    loc = plticker.MultipleLocator(base=200.0)  # this locator puts ticks at regular intervals
    bins = range(3000, 4450, 50)

    x1 = list(df_ml_jh_plot[df_ml_jh_plot['Machine learning type '] == 'S-type zircon']['Age（Ma)'])
    x2 = list(df_ml_jh_plot[df_ml_jh_plot['Machine learning type '] == 'I-type zircon']['Age（Ma)'])
    f, ax = plt.subplots(figsize=(4, 3))
    plt.hist(
        [x1, x2],
        bins=bins,
        stacked=True,
        #     normed=True,
        color=colors_JH,
        #     alpha=0.5,
        edgecolor="w",
        linewidth=0.3,
        label=ml_ziron_type
    )

    plt.legend(loc='upper right')
    plt.xlabel('Age(Ma)', fontsize=13)
    plt.ylabel('Frequency', fontsize=13)
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(plticker.MultipleLocator(base=20))
    plt.xlim(3000, 4400)
    ax.tick_params(axis="y", direction="in")
    ax.tick_params(axis="x", direction="in")
    # plt.tight_layout()
    plt.savefig(figPath + 'ML_JH_stacked_hist0104.png')
    plt.show()
