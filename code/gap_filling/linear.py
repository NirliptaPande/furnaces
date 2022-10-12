#In this file, we will use linear interpolation across a month to fill in the missing data, where needed, or we can fill in zeros if needed like in ignition
import numpy as np
#from mpl_toolkits.basemap import Basemap
#import matplotlib.pyplot as plt
from multiprocessing import Pool
import gc
from math import nan
def fill_mean(idx):
      var = spatial[:,0:,0:,idx]
      var_mean = np.nanmean(var,axis = 1)
      np.nan_to_num(var_mean,nan = 0,copy = False)
      for i in range(12):
            temp = var[:,0:,i]
            #temp.shape = (244243, 9) (pixels, years)
            temp_mean = var_mean[:,i]
            temp_mean = np.reshape(temp_mean,(-1,1))
            np.nan_to_num(temp_mean,copy= False,nan=0)
            #temp_mean.shape = (244243,1)
            inds = np.where(np.isnan(temp))
            temp[inds] = np.take(temp_mean,inds[1])
            #I have absolutely no idea what this is, picked it up from stackoverflow
            #I don't think this line is correct, or will work for 3D
            #the above line is supposed to fill in the mean values for each varible, 
            # but if you can fill in 3D, you should be able to fill in 4D, you shouldn't need a loop
            var[:,0:,i] = temp
      return var

def fill_linear(idx):
      var = np.copy(spatial[:,0:,0:,idx])
      for i in range(12):
            arr = var[:,0:,i]
            arr_int = np.zeros_like(arr,dtype = 'float')
            test_sum = np.isnan(arr).sum(axis =0)
            for i in range(arr.shape[1]):
                  temp = arr[:,i]
                  test = test_sum[i]
                  #print(test)
                  if (test == arr.shape[0]):#If all are false along a column
                        arr_int[:,i]=0
                  else:
                        nans = np.isnan(temp)
                        x = lambda z:z.nonzero()[0]
                        temp[nans] = np.interp(x(nans),x(~nans),temp[~nans])
                        arr_int[:,i]= temp
            var[:,0:,i] = arr_int
      return var      
      
train = np.load('../../data/train_sans_ocean.npy')
pixels = train.shape[0]//108
temp = np.reshape(train,(pixels,-1,train.shape[1]))
lat = temp[:,0][:,0]
lon = temp[:,0][:,1]
del(temp)
gc.collect()
spatial = np.reshape(train,(pixels,9,12,train.shape[1]))#converts the array into 4D, first is all pixels, second is years,third is months and fourth is features
#The array is structured as (pixels,years,months,features)
#Index(['lat 0', 'lon 1', 'time 2', 'agb 3', 'pft_fracCover 4', 'sm 5', 'pftCrop 6',
       #'pftHerb 7', 'pftShrubBD 8', 'pftShrubNE 9', 'pftTreeBD 10', 'pftTreeBE 11',
       #'pftTreeND 12', 'pftTreeNE 13', 'GDP 14', 'ign 15', 'Distance_to_populated_areas 16',
       #'fPAR 17', 'LAI 18', 'NLDI 19', 'vod_K_anomalies 20', 'FPAR_12mon 21', 'LAI_12mon 22',
       #'Vod_k_anomaly_12mon 23', 'FPAR_06mon 24', 'LAI_06mon 25', 'Vod_k_anomaly_06mon 26',
       #'WDPA_fracCover 27', 'dtr 28', 'pet 29', 'tmx 30', 'wet 31', 'Biome 32', 'precip 33',
       #'Livestock 34', 'road_density 35', 'topo 36', 'pop_density 37'],
      #dtype='object')
#lat, lon, time don't need any interpolation
#
del(train)
gc.collect()
"""
I think in the first part, I'll fill in the data with the monthly mean, in all NaNs
"""
idxs = range(3,38,1)
with Pool() as pool:
     mean = pool.map( fill_mean, idxs)
pool.close()
pool.join()
np.save("../../data/month_mean.npy",mean)
del(mean)
gc.collect()
"""
0,1,2 don't need any interpolation,
I'll fill in all the rest with mean of a certain pixel over a certain month, 
if a certain pixel over a certain month is all NaN, I'll fill it with zeros
I'll fill NaNs with zeros in ignition
"""
idxs = range(3,38,1)
with Pool() as pool:
      inter = pool.map(fill_linear,idxs)
pool.close()
pool.join()
np.save("../../data/linear_month.npy",inter)