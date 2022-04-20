import speasy as spz
import os
import sys
import pickle as pkl
from datetime import datetime, timedelta
import numpy as np

from speasy_utils import printProgressBar

# to improve reliability of the connection to AMDA never request
# more than a month of data
def amda_get_data(param_id, start, stop, verbose=True):
    block_size = timedelta(days=30)
    blocks = []
    curr_t = start
    T = (stop - start).total_seconds()
    while curr_t < stop:
        if verbose:
            t = (curr_t - start).total_seconds()
            it = int(100. * (t/T))
            #sys.stdout.write("{:.2f} ".format(100. * (t/T)))
            printProgressBar(it, 100, prefix = f"Param:{param_id}", length=50)
        if curr_t + block_size <= stop:
            p = spz.get_data(param_id, curr_t, curr_t + block_size)
        else:
            p = spz.get_data(param_id, curr_t, stop)
        if p is not None:
            # fill values
            p.data = np.where(p.data <= -1.e31, np.nan, p.data)
            blocks.append(p)
        curr_t += block_size
    # merge blocks
    if len(blocks)==0:
        return None
    if len(blocks)==1:
        return blocks[0]
    for i in range(1,len(blocks)):
        if blocks[i] is None:
            continue
        blocks[0].time = np.hstack((blocks[0].time,blocks[i].time))
        blocks[0].data = np.vstack((blocks[0].data,blocks[i].data))
    # duplicate times
    _,unq_indx = np.unique(blocks[0].time, return_index=True)
    blocks[0].time = blocks[0].time[unq_indx]
    blocks[0].data = blocks[0].data[unq_indx]
    return blocks[0]
    
def get_time_shifted(param_id, start, stop, delta_t):
    p = amda_get_data(param_id, start - delta_t, stop - delta_t)
    p.time += delta_t.total_seconds()
    return p

def get_parameter_data(parameter_ids, start, stop, data_filename, shifts={}):
    parameter_data={}
    if os.path.exists(data_filename):
        parameter_data = pkl.load(open(data_filename,"rb"))
    else:
        for p in parameter_ids:
            if p in shifts:
                parameter_data[p] = get_time_shifted(f"amda/{p}",start,stop,shifts[p])
            else:
                parameter_data[p] = amda_get_data(f"amda/{p}", start, stop)
    if not all([(p in parameter_data) for p in parameter_ids]):
        os.system(f"rm -f {data_filename}")
        print("Missing data. Deleting data file and downloading again.")
        return get_parameter_data(parameter_ids, start, stop, data_filename, shifts=shifts)
    pkl.dump(parameter_data, open(data_filename,"wb"))
    return parameter_data

def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]
def clean_data(y):
    nans, x = nan_helper(y)
    y[nans] = np.interp(x(nans), x(~nans), y[~nans])
    return y

def clean_parameter_data(param_data):
    for k in param_data:
        if np.any(np.isnan(param_data[k].data)):
            new_data = clean_data(param_data[k].data)
            param_data[k].data = new_data
    return param_data

from datetime import timedelta
def get_interpolation_bounds(parameter_data):
    bounds = [None, None]
    for p in parameter_data.values():
        t = p.time
        if bounds[0] is None or np.min(t)>bounds[0]:
            bounds[0] = np.min(t)  #timedelta(seconds=1)
        if bounds[1] is None or np.max(t)<bounds[1]:
            bounds[1] = np.max(t)  #timedelta(seconds=1)
    return [datetime.utcfromtimestamp(e) for e in bounds]

def interpolate_data(parameter_data, bounds, delta_t, freq="60S"):
    from scipy.interpolate import interp1d
    import pandas as pd
    interp_data = {}

    for k in parameter_data:
        interp_data[k] = interp1d(parameter_data[k].time, parameter_data[k].data, axis=0)
    #t = pd.date_range(bounds[0]+delta_t, bounds[1]-delta_t, freq=freq).astype(np.int64)/1e9
    t = pd.date_range(bounds[0]+delta_t, bounds[1]-delta_t, freq=freq).view(np.int64)/1e9

    interpolated_data = {k: interp_data[k](t) for k,v in parameter_data.items()}
    return t, interpolated_data


#def get_parameter_data(parameter_ids, start, stop, data_filename, shifts={}):
def get_interpolated_data(parameter_ids, start, stop, data_filename, shifts={}, freq="60S", delta_t=timedelta(seconds=0)):
    parameter_data = get_parameter_data(parameter_ids, start, stop, data_filename, shifts=shifts)

    parameter_data = clean_parameter_data(parameter_data)
    bounds = get_interpolation_bounds(parameter_data)

    # interpolate
    t, interpolated_data = interpolate_data(parameter_data, bounds, delta_t, freq=freq)
    
    return t, interpolated_data, bounds


