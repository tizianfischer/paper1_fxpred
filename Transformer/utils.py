import os
import copy
import numpy as np
import pandas as pd
from typing import List


def data_read_dict(fp):
    return {
        k.replace('.csv', '').lower() : pd.read_csv(os.path.join(fp, k), index_col=0, parse_dates=True)
        for k in sorted(os.listdir(fp))
    }

def data_read_concat(fp):
    data = data_read_dict(fp)
    for k, v in data.items():
       v.loc[:, 'fx'] = k
       v.reset_index(inplace=True)
    d = pd.concat([v for k, v in data.items()])
    return d

def data_merge(data):
    d = copy.deepcopy(data)
    tmp = None
    for k, v in d.items():
        v.columns = [k + '___' + i for i in v.columns.to_list()]
        if tmp is None:
            tmp = v
        else:
            tmp = pd.merge(tmp, v, left_index=True, right_index=True, how='outer')
    return tmp


def get_fx_and_metric_data(
    *,
    pct_change:bool=True,
    dtype:np.float=None
) -> pd.DataFrame:
    """Gets the FX spot rates and combines data with metrics

    Args:
        pct_change (bool, optional): Returns percantage change. Defaults to True.
        dtype (numpy.float, optional): data type of data, options 'numpy.floatX'. Defaults to None.

    Returns:
        pd.DataFrame: Spot rates and metrics in one pandas.DataFrame
    """

    path = 'data/10min Dataset Spot.csv'
    df = pd.read_csv(path, delimiter=';')
    df['Dates'] = pd.to_datetime(df['Dates'], format='%d.%m.%y %H:%M')
    df.set_index('Dates', inplace=True)
    df = df.asfreq('600S').ffill()
    if pct_change:
        df = df.pct_change()[1:]
    assert len(set(np.diff(df.index.values))) == 1

    FX_Fundamentals_path = 'data/10min Dataset Rest.csv'
    df2 = pd.read_csv(FX_Fundamentals_path, delimiter=';')
    df2.replace(to_replace=0, method='ffill', inplace=True) # Replace 0 to avoid dividing by 0 later on
    df2.drop('UXA1 Comdty Trade Open', axis=1, inplace=True)
    df2['Dates'] = pd.to_datetime(df2['Dates'], format='%d.%m.%y %H:%M')
    df2.sort_values('Dates', inplace=True)
    df2.sort_values('Dates')
    df2.index = df2['Dates']
    # df2

    df3 = pd.merge(df, df2, left_index=True, right_index=True)
    df3['Dates'] = pd.to_datetime(df3['Dates'], format='%d.%m.%y %H:%M')
    df3.index = df3['Dates']
    # df3

    df_metrics = data_merge(data_read_dict('data/bbg/'))
    df_metrics.shape
    # excluding eurgbp for now
    df_metrics = df_metrics.loc[:, [i for i in df_metrics.columns if i.split('___')[0].lower() != 'eurgbp']]

    df_merged = pd.merge(df3, df_metrics, left_index=True, right_index=True, how='outer')
    df_merged = df_merged.loc[df_merged.index <= max(df3.Dates),:]
    df = df_merged[:]
    df.drop('Dates', axis=1, inplace=True)
    df = df.asfreq('600S').ffill()
    df = df.astype(dtype)
    df[df == np.infty] = 0 
    df[df == -np.infty] = 0
    # df.dropna(how='all', axis=0, inplace=True) # Drop all rows with NaN values"
    # TODO: fixing still missing values in metric data
    df.fillna(0, inplace=True)
    df = df.loc[(df.index >= '2020-11-01') & (df.index < '2021-08-01'), :]
    del df_merged, df2, df3
    return df


def get_fx_and_metric_data_wo_weekend(
    *,
    pct_change:bool=True,
    dtype:np.float=None
) -> pd.DataFrame:
    """Gets the FX spot rates and combines data with metrics, without missing values on weekends (and bank holidays).

    Args:
        pct_change (bool, optional): Returns percantage change. Defaults to True.
        dtype (numpy.float, optional): data type of data, options 'numpy.floatX'. Defaults to None.

    Returns:
        pd.DataFrame: Spot rates and metrics in one pandas.DataFrame
    """

    path = 'data/10min Dataset Spot.csv'
    df = pd.read_csv(path, delimiter=';')
    df['Dates'] = pd.to_datetime(df['Dates'], format='%d.%m.%y %H:%M')
    df.set_index('Dates', inplace=True)
    # df = df.asfreq('600S')
    if pct_change:
        df = df.pct_change()[1:]
    # assert len(set(np.diff(df.index.values))) == 1

    FX_Fundamentals_path = 'data/10min Dataset Rest.csv'
    df2 = pd.read_csv(FX_Fundamentals_path, delimiter=';')
    df2.replace(to_replace=0, method='ffill', inplace=True) # Replace 0 to avoid dividing by 0 later on
    df2.drop('UXA1 Comdty Trade Open', axis=1, inplace=True)
    df2['Dates'] = pd.to_datetime(df2['Dates'], format='%d.%m.%y %H:%M')
    df2.sort_values('Dates', inplace=True)
    df2.sort_values('Dates')
    df2.index = df2['Dates']
    # df2

    df3 = pd.merge(df, df2, left_index=True, right_index=True)
    df3['Dates'] = pd.to_datetime(df3['Dates'], format='%d.%m.%y %H:%M')
    df3.index = df3['Dates']
    # df3

    df_metrics = data_merge(data_read_dict('data/bbg/'))
    df_metrics.shape
    # excluding eurgbp for now
    df_metrics = df_metrics.loc[:, [i for i in df_metrics.columns if i.split('___')[0].lower() != 'eurgbp']]
    df_merged = pd.merge(df3, df_metrics, left_index=True, right_index=True, how='outer')
    
    # Deleting all rows that have missing values in df and df2 columns
    df_merged = df_merged.loc[~df_merged.loc[:, set(df.columns.append(df2.columns))].isna().all(axis=1)]

    # Deleting all columns with duplicated data that occur on the weekend
    #(np.diff(df_merged.loc[:, df.columns], prepend=-9999) == 0).mean(axis=1) < 0.95
    duplicates = df_merged.loc[:, df.columns].duplicated()
    weekend = [i.weekday() in [5, 6] for i in df_merged.index]
    # sum(duplicates)
    # sum(weekend)
    # sum(duplicates & weekend)
    # sum(duplicates | weekend)
    df_merged = df_merged.loc[~(duplicates & weekend), :]

    df = df_merged[:]
    df.drop('Dates', axis=1, inplace=True)
    # df = df.asfreq('600S')
    df = df.ffill()
    df = df.loc[(df.index >= '2020-11-01') & (df.index < '2021-08-01'), :]
    #TODO: There will still be NA values (metric values) in the beginning, possible fixes:
    df = df.bfill()  # back fill
    # df.dropna(how='all', axis=0, inplace=True)  # Drop all rows with NaN values"
    # df.fillna(0, inplace=True)  # impute all NaNs with 0
    df = df.astype(dtype)    
    del df_merged, df2, df3
    return df
