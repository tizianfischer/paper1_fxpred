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

if platform.node() in ['mstp3', 'msbq']:
    os.chdir('/home/ms/github/fxpred/data')

fp_raw = 'bbg_raw'
fp_clean = 'bbg'

if not os.path.isdir(fp_clean):
    os.makedirs(fp_clean)


for fp in tqdm.tqdm(os.listdir(fp_raw)):
    x = pd.read_excel(os.path.join(fp_raw, fp))
    x2 = pd.read_excel(os.path.join(fp_raw + '2', fp))
    x3 = pd.read_excel(os.path.join(fp_raw + '3', fp))

    # delete all columns without input
    x = x.loc[:, ~x.isna().all(axis=0)]
    x2 = x2.loc[:, ~x2.isna().all(axis=0)]
    x3 = x3.loc[:, ~x3.isna().all(axis=0)]
    fx_name = x.columns[0].replace(' Curncy', '')
    fx_name2 = x2.columns[0].replace(' Curncy', '')
    fx_name3 = x3.columns[0].replace(' Curncy', '')

    # rename columns
    x.columns = x.iloc[1, :]
    x2.columns = x2.iloc[1, :]
    x3.columns = x3.iloc[1, :]

    # delete first 2 lines
    x = x.iloc[2:, :]
    x2 = x2.iloc[2:, :]
    x3 = x3.iloc[2:, :]

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
    res3 = clean(x3)
    assert res2.shape[0] != 0, f'{fp} is empty'
    assert res3.shape[0] != 0, f'{fp} is empty'
    d_max = res.index.max()
    d_max2 = res2.index.max()

    # test = pd.merge(res, res2, how='inner', left_index=True, right_index=True)
    # dd = pd.DataFrame.from_dict(
    #     {c: (np.abs(res.loc[test.index, c] - res2.loc[test.index, c])).mean() for c in res.columns},
    #     orient='index'
    # )
    # dd.plot.barh()
    # plt.show()
    # d_max = res.merge(res2, how='inner', left_index=True, right_index=True).index.max()
    # res.merge(res2.loc[res2.index > d_max], how='outer', left_index=True, right_index=True, indicator='False')

    res = pd.concat([res, res2.loc[res2.index > d_max], res3.loc[res3.index > d_max2]])
    res.to_csv(os.path.join(fp_clean, fx_name + '.csv'))
    