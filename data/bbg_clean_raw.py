#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 05:53:03 2021

@author: ms
"""
import os
import tqdm
import platform
import pandas as pd
pd.set_option('display.max_columns', None)

if platform.node() in ['mstp3']:
    os.chdir('/home/ms/github/fxpred/data')

fp_raw = 'bbg_raw'
fp_clean = 'bbg'

if not os.path.isdir(fp_clean):
    os.mkdirs(fp_clean)

for fp in tqdm.tqdm(os.listdir(fp_raw)):
    if os.path.isfile(os.path.join(fp_clean, fp.upper().replace('XLSX', 'csv'))):
        continue
    # read data from xlsx
    x = pd.read_excel(os.path.join(fp_raw, fp))

    # delete all columns without input
    x = x.loc[:, ~x.isna().all(axis=0)]
    fx_name = x.columns[0].replace(' Curncy', '')

    # checking that all dates are same
    con = False
    for c in x.loc[:, x.iloc[1, :] == 'Dates']:
        # assert (x.loc[:, fx_name + ' Curncy'] == x.loc[:, c]).all(), fx_name
        if not (x.loc[:, fx_name + ' Curncy'] == x.loc[:, c]).all():
            con = True
    if con:
        print(fx_name)
        continue
    # set index to Dates
    x.index = x.loc[:, x.iloc[1, :] == 'Dates'].iloc[:, 0]

    # deleting all Dates columns
    x = x.loc[:, x.iloc[1, :] != 'Dates']

    # rename columns
    x.columns = x.iloc[1, :]
    x = x.iloc[2:, :]

    x.to_csv(os.path.join(fp_clean, fx_name + '.csv'))
