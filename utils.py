import os
import copy
import pandas as pd

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
