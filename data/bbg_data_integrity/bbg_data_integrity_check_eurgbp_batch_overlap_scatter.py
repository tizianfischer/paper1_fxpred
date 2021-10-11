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



x = pd.read_excel(os.path.join(fp_raw, 'eurgbp.xlsx'))
x2 = pd.read_excel(os.path.join(fp_raw + '2', 'batch2_4.xlsx'), sheet_name='eurgbp')

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

time_overlap = res.index.intersection(res2.index)

k = res.columns[22]

path = 'bbg_data_integrity/eurgbp_batch_overlap_scatter'
for k in res.columns:
    x = res.loc[time_overlap, k]
    y = res2.loc[time_overlap, k]
    time_overlap_valid = x.index[~x.isna()].intersection(y.index[~y.isna()])
    plt.scatter(
        x[time_overlap_valid],
        y[time_overlap_valid]
    )
    # plt.show()
    plt.savefig(os.path.join(path, f'eurgbp_{k}.png'))
    plt.close()

# %% 
data = dict()
for f in os.listdir('bbg'):
    if not ('EUR' in f or 'GBP' in f) or 'JPY' in f:
        continue
    x = pd.read_csv(os.path.join('bbg', f))
    x.Dates = pd.to_datetime(x.Dates)
    x = x.set_index('Dates')
    data.update({f.lower().rstrip('.csv'): x})

path = 'bbg_data_integrity/eur_or_gbp_metrics'
for c in data['eurgbp'].columns:
    for k, v in data.items():
        if k == 'eurgbp' and c in [
            'ATR', 'EMAVG', 'FEAR_GREED', 'MAE_LOWER', 'MAE_MIDDLE',\
            'MAE_UPPER', 'MAO_DIFF', 'MAOsc', 'MAO_SIGNAL', 'MAX', \
            'MIN', 'MM_RETRACEMENT', 'MOMENTUM', 'SMAVG', 'VMAVG', \
            'WMAVG'
        ]:
            # v1 = v.loc[v.index < '2021-04-01', c].iloc[-1]
            # v2 = v.loc[v.index >= '2021-04-01', c].iloc[0]
            v.loc[v.index < '2021-04-01', c] /= 100
            plt.plot(v.loc[:, c], label=k)
        else:
            plt.plot(v.loc[:, c], label = k)
    # plt.show()
    plt.legend()
    plt.savefig(os.path.join(path, f'{c}.png'))
    plt.close()

