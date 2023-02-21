import numpy as np
from multiprocessing import Pool
import gc
import warnings
warnings.filterwarnings("ignore")

train = np.load("../../data/new/train_new.npy")
timesteps = 108
pixels = train.shape[0]//timesteps
# temp = np.reshape(train,(pixels,-1,train.shape[1]))
# lat = temp[:,0][:,0]
# lon = temp[:,0][:,1]
# del(temp)
gc.collect()
train = np.reshape(train, (pixels, 9, 12, train.shape[1]))
# converts the array into 4D, first is all pixels, second is years,third is months and fourth is features
# The array is structured as (pixels,years,months,features)
"""['lat 0', 'lon 1', 'time 2', 'road_density 3', 'agb 4', 'sm 5', 'pftCrop 6', 'pftHerb 7',
       'pftShrubBD 8', 'pftShrubNE 9', 'pftTreeBD 10', 'pftTreeBE 11', 'pftTreeND 12',
       'pftTreeNE 13', 'GDP_per_capita_PPP 14', 'ign 15', 'faPAR 16', 'LAI 17',
       'vod_K_anomalies 18', 'primn 19', 'pet 20', 'wet 21', 'precip 22', 'pastr 23', 'Band1 24',
       'pop_density 25']"""
# Fill in zeros for ign, faPAR, LAI, vod_k_anamolies, pop_density
# 15,16,17,18,25
idxs = [15, 16, 17, 18, 25]
temp = np.zeros_like(train)
for idx in idxs:
    var = train[:, 0:, 0:, idx]
    var = np.nan_to_num(var, nan=0)
    if idx == 25:
        var[var < 0] = 0
    temp[:, 0:, 0:, idx] = var
    train[:, 0:, 0:, idx] = var
idxs = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24]
# global nan_tuple
# nan_tuple = tuple()


def custom_fill(idx_new):
    var = train[:, 0:, 0:, idx_new]
    var_med = np.copy(var)
    var_mean = np.nanmean(var, axis=1)
    var_median = np.nanmedian(var, axis=1)
    # Check back on it once project machine is empty
    # empty = np.where(np.isnan(var_mean))
    # nan_tuple = (*nan_tuple,*empty)
    var_mean = np.nan_to_num(var_mean, nan=0)
    var_med = np.nan_to_num(var_med, nan=0)
    for i in range(var.shape[1]):
        var[:, i] = np.nan_to_num(var[:, i], nan=var_mean)
        var_med[:, i] = np.nan_to_num(var[:, i], nan=var_median)
    temp[:, 0:, 0:, idx_new] = var
    train[:, 0:, 0:, idx_new] = var_med


with Pool() as pool:
    pool.map(custom_fill, idxs)
pool.close()
pool.join()

train_x = np.load("../../data/train_monthmean.npy")
train_x = np.reshape(train_x,(-1,train_x.shape[-1]))
t_max = train_x[:,(16,28,30,34)]
del(train_x)
gc.collect()

temp = temp.reshape((-1,temp.shape[-1]))
train = train.reshape((-1,train.shape[-1]))
temp = np.hstack((temp,t_max))
train = np.hstack((train,t_max))

col_names = ['lat 0', 'lon 1', 'time 2', 'road_density 3', 'agb 4', 'sm 5', 'pftCrop 6', 'pftHerb 7',
       'pftShrubBD 8', 'pftShrubNE 9', 'pftTreeBD 10', 'pftTreeBE 11', 'pftTreeND 12',
       'pftTreeNE 13', 'GDP_per_capita_PPP 14', 'ign 15', 'faPAR 16', 'LAI 17',
       'vod_K_anomalies 18', 'primn 19', 'pet 20', 'wet 21', 'precip 22', 'pastr 23', 'Band1 24',
       'pop_density 25', 'Distance_to_populated_areas 26', 'Diurnal temperature range 27', 'temp_max 28', 'Livestock 29']

np.save("../../data/new/custom_mean.npy", temp)
np.save("../../data/new/custom_median.npy", train)
np.save("../../data/new/column_names.npy", col_names)