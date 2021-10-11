#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 05:53:03 2021

 @author: ms
"""
import os
import tqdm
import platform
import numpy as np
import pandas as pd
from openpyxl import load_workbook
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt

if platform.node() in ['mstp3', 'msbq', 'msdai']:
    os.chdir('/home/ms/github/fxpred/data')

fp_raw = 'bbg_raw'
fp_clean = 'bbg'

if not os.path.isdir(fp_clean):
    os.makedirs(fp_clean)

data = {}
for fp in tqdm.tqdm(os.listdir(fp_raw)):
    x = pd.read_excel(os.path.join(fp_raw, fp))
    x2 = pd.read_excel(os.path.join(fp_raw + '2', fp))

    # delete all columns without input
    x = x.loc[:, ~x.isna().all(axis=0)]
    x2 = x2.loc[:, ~x2.isna().all(axis=0)]
    fx_name = x.columns[0].replace(' Curncy', '')
    fx_name2 = x2.columns[0].replace(' Curncy', '')

    # rename columns
    x.columns = x.iloc[1, :]
    x2.columns = x2.iloc[1, :]

    # delete first 2 lines
    x = x.iloc[2:, :]
    x2 = x2.iloc[2:, :]

    # checking that all dates are same
    def clean(x):
        dates = np.where(x.columns == 'Dates')[0]
        if np.all([(x.iloc[:, dates[0]] == x.iloc[:, i]).all() for i in dates]):
            # set index to Dates
            x.index = x.iloc[:, dates[0]]
            # deleting all Dates columns
            res = x.loc[:, x.columns != 'Dates']
            res.columns.name = None
        else:
            res = None
            for i, c in enumerate(dates):
                if i < len(dates) - 1:
                    c_end = dates[i + 1]
                else:
                    c_end = x.shape[1]
                tmp = x.iloc[2:, c:c_end].dropna(how='all')
                tmp.index = pd.to_datetime(tmp.Dates)
                tmp = tmp.drop('Dates', axis=1)
                # tmp = pd.DataFrame(tmp[:, 1:], index=tmp.Dates)
                # tmp = pd.DataFrame(x.iloc[2:, c + 1:c_end]#, index = pd.to_datetime(x.iloc[2:, c])
                # )
                # tmp.index = pd.to_datetime(x.iloc[2:, c])
                # tmp = tmp.reindex(pd.to_datetime(x.iloc[2:, c]), axis=0)
                # tmp.isna()
                # tmp.index.isna()
                tmp = tmp.asfreq('600S')
                tmp.columns.name = None
                if res is None:
                    res = tmp
                else:
                    res = pd.merge(res, tmp, how='outer', left_index=True, right_index=True)
                del tmp
        return res
    res = clean(x)
    res2 = clean(x2)
    data.update({fp.split('.')[0] + '_1': res})
    data.update({fp.split('.')[0] + '_2': res2})

# data.keys()
# for c in data['usdchf_1'].columns:
#     for l in [i for i in data.keys() if '_1' in i]:
#         data[l].loc[:, c].plot()
#         data[l[:-1] + '2'].loc[:, c].plot()
#     plt.show()

# %% plot each
if not os.path.exists('bbg_data_integrity/data_vis_per_fx_pair'):
    os.makedirs('bbg_data_integrity/data_vis_per_fx_pair')
for l in tqdm.tqdm(sorted(set(i[:-2] for i in data.keys()))):
    # if os.path.exists(f'bbg_data_integrity/data_vis_per_fx_pair/{l}.png'):
    #    continue
    fig, ax = plt.subplots(nrows=7, ncols=6, clear=True, figsize=[16, 12])
    # fig = plt.figure(figsize=[12.8, 9.6])
    fig.suptitle(l)
    dmax = data[l + '_1'].index.max()
    cols = set(data[l + '_1'].columns)
    # cols2 = set(data[l + '_2'].columns)
    # assert cols - cols2 == set()
    # assert cols2 - cols == set()
    for i, c in enumerate(sorted(cols)):
        # fig.subplot(7, 6, 1 + i)
        # frame1 = plt.gca()
        # plt.gca()plt.gca()
        # data[l + '_2'].query(f'index <= "{dmax}"').loc[:, c].plot(label=l + '2', ax=ax[i // 6, i % 6], c='grey')
        data[l + '_1'].loc[:, c].plot(label=l + '1', ax=ax[i // 6, i % 6], c='tab:blue')
        data[l + '_2'].query(f'index > "{dmax}"').loc[:, c].plot(label=l + '2', ax=ax[i // 6, i % 6], c='tab:green')
        data[l + '_2'].query(f'index <= "{dmax}"').loc[:, c].plot(label=l + '2', ax=ax[i // 6, i % 6], c='tab:gray', linewidth=0.8, linestyle=':')
        ax[i // 6, i % 6].axes.set_xlabel('')
        if i < 36:
            ax[i // 6, i % 6].get_xaxis().set_ticks([])
        # plt.xlabel('')
        ax[i // 6, i % 6].set_title(c)
    plt.tight_layout()
    plt.savefig(f'bbg_data_integrity/data_vis_per_fx_pair/{l}.png', bbox_inches='tight')
    # plt.show()
    plt.close(fig)

# %% overlapping batch data
data_join = {
    l[:-2]: 
    pd.merge(data[l], data[l[:-1] + '2'], how='inner', left_index=True, right_index=True)
    for l in [i for i in data.keys() if '_1' in i]
}
#%% Plot difference of metrics in overlapping time
if not os.path.exists('bbg_data_integrity/difference_overlap'):
    os.mkdir('bbg_data_integrity/difference_overlap')
for i, c in tqdm.tqdm(enumerate(sorted(set(i[:-2] for i in data_join['eurusd'].columns)))):
    fig = plt.figure(i)
    for l in data_join.keys():
        if 'eurgbp': continue
        (data_join[l][c + '_x'] - data_join[l][c + '_y']).plot(label=l)
    plt.xlabel('Time')
    plt.ylabel(f'Batch1 {c} - Batch2 {c}')
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    # plt.show()
    plt.savefig(f'bbg_data_integrity/difference_overlap/{c}.png', bbox_inches='tight')
    plt.close(fig)

#%% Plot difference of metrics in overlapping time without eurgbp
if not os.path.exists('bbg_data_integrity/difference_overlap_wo_eurgbp_audusd'):
    os.mkdir('bbg_data_integrity/difference_overlap_wo_eurgbp_audusd')
for i, c in tqdm.tqdm(enumerate(sorted(set(i[:-2] for i in data_join['eurusd'].columns)))):
    fig = plt.figure(i)
    for l in data_join.keys():
        if l in ['eurgbp', 'audusd']: continue
        (data_join[l][c + '_x'] - data_join[l][c + '_y']).plot(label=l)
    plt.xlabel('Time')
    plt.ylabel(f'Batch1 {c} - Batch2 {c}')
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    # plt.show()
    plt.savefig(f'bbg_data_integrity/difference_overlap_wo_eurgbp_audusd/{c}.png', bbox_inches='tight')
    plt.close(fig)

# %% Plot comparison of eurgpb and audusd
if not os.path.exists('bbg_data_integrity/comparison_eurgbp_audusd'):
    os.mkdir('bbg_data_integrity/comparison_eurgbp_audusd')
for i, c in tqdm.tqdm(enumerate(set(i[:-2] for i in data_join['eurusd'].columns))):
    fig = plt.figure(i)
    for l in sorted(set(i[:-2] for i in data.keys())):
        if l not in ['eurgbp', 'audusd']: continue
        data[l + '_1'].loc[:, c].plot(label=l + '1')
        data[l + '_2'].loc[:, c].plot(label=l + '2')
    plt.xlabel('Time')
    plt.ylabel(c)
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    # plt.show()
    plt.savefig(f'bbg_data_integrity/comparison_eurgbp_audusd/{c}.png', bbox_inches='tight')
    plt.close(fig)
