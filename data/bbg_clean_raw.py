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
pd.set_option('display.max_columns', None)

if platform.node() in ['mstp3']:
    os.chdir('/home/ms/github/fxpred/data')

fp_raw = 'bbg_raw'
fp_clean = 'bbg'

if not os.path.isdir(fp_clean):
    os.mkdirs(fp_clean)

for fp in tqdm.tqdm(os.listdir(fp_raw)):
    # if os.path.isfile(os.path.join(fp_clean, fp.upper().replace('XLSX', 'csv'))):
    #     continue
    # read data from xlsx
    x = pd.read_excel(os.path.join(fp_raw, fp))

    # delete all columns without input
    x = x.loc[:, ~x.isna().all(axis=0)]
    fx_name = x.columns[0].replace(' Curncy', '')

    # rename columns
    x.columns = x.iloc[1, :]

    # delete first 2 lines
    x = x.iloc[2:, :]

    # checking that all dates are same
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
            tmp = pd.DataFrame(x.iloc[2:, c + 1:c_end])
            tmp.index = pd.to_datetime(x.iloc[2:, c])
            tmp.reindex(pd.to_datetime(x.iloc[2:, c]), axis=0)
            tmp.columns.name = None
            if res is None:
                res = tmp
            else:
                res = pd.merge(res, tmp, left_index=True, right_index=True)
            del tmp
    res.to_csv(os.path.join(fp_clean, fx_name + '.csv'))