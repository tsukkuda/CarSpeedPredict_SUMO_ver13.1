import numpy as np
import pandas as pd
import math
from joblib import Parallel, delayed
import ReadData.tqdm_joblib as tj

#[x]
def resolutionChange_func(speed,volume,window):
    def summerize_list(speed,volume,window_size):
        res = []
        #math.ceil : float型の小数点以下切り上げ
        for i in range(math.ceil(len(speed)-1/window_size)-2):
            slice_begin = window_size * i + 1
            slice_last  = window_size * (i + 1) + 1
            speed_nd  = np.array(speed[slice_begin:slice_last])
            volume_nd = np.array(volume[slice_begin:slice_last])
            doted_speed = np.nansum(speed_nd*volume_nd)
            sum_volume  = np.nansum(volume_nd)
            res.append(np.average(speed_nd))

        return res

    list_speed  = list(speed)
    list_volume = list(volume)

    dict_summerized = {}
    dict_summerized[speed.name] = summerize_list(list_speed,
                                                    list_volume,
                                                    window)

    res = pd.DataFrame(dict_summerized)
    return res

#[x]
def split_data(i,df,window):
    loc_val = int(len(df.columns) - 2)
    res = {}

    for i in range(loc_val):
        changed_df = resolutionChange_func(speed  = df.iloc[:,2+i],
                                            volume = df.iloc[:,2+i],
                                            window = window)

        changed_dict = changed_df.to_dict(orient="list")
        res.update(changed_dict)

    return pd.DataFrame(res)

#[x]
def split_data_func(datalist,window):
       
    with tj.tqdm_joblib(len(datalist)):
        origin = Parallel(n_jobs = -1)([delayed(split_data)(i,datalist[i].iloc[:,1:],window) for i in range(0,len(datalist))])

    return origin
