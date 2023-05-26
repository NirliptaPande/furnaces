#Use this instead of mean.py to find the mean, idk why it doesn't work 
import numpy as np
from multiprocessing import Pool
import gc
import warnings
warnings.filterwarnings("ignore")
"""
This subsection gap fills the train data and stores it inside furnaces/data/ as train_monthmean.py
"""
train = np.load('../../data/train_sans_ocean.npy')
timesteps = 108
# test = np.load('../../data/test_sans_ocean.npy')
# timesteps = 57
pixels = train.shape[0]//timesteps
# temp = np.reshape(train,(pixels,-1,train.shape[1]))
# lat = temp[:,0][:,0]
# lon = temp[:,0][:,1]
# del(temp)
# gc.collect()
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
for idx in range(spatial.shape[3]):
    var = spatial[:,0:,0:,idx]
    var_mean = np.nanmean(var,axis = 1)
    np.nan_to_num(var_mean,nan = 0,copy = False)
    inds = np.where(np.isnan(var))
    spatial[:,0:,0:,idx][inds] = var_mean[(inds[0],inds[2])]
np.save("../../data/train_monthmean.npy",spatial)
del(spatial)
"""
This deals in test data, but do remove the last 3 slices of the data when testing for the results
"""
test = np.load('../../data/test_sans_ocean.npy')
timesteps = 57
pixels = test.shape[0]//timesteps
temp = np.reshape(test,(pixels,-1,test.shape[1]))
block = np.zeros((pixels,3,test.shape[1]))
block[:] = np.nan
temp = np.append(temp,block,axis = 1)
spatial  = np.reshape(temp,(pixels,5,12,test.shape[1]))#it goes pixels, years, months, features, I am basically splitting up 60
del(test)
del(block)
del(temp)
gc.collect()
for idx in range(spatial.shape[3]):
    var = spatial[:,0:,0:,idx]
    var_mean = np.nanmean(var,axis = 1)
    np.nan_to_num(var_mean,nan = 0,copy = False)
    inds = np.where(np.isnan(var))
    spatial[:,0:,0:,idx][inds] = var_mean[(inds[0],inds[2])]
np.save("../../data/test_monthmean.npy",spatial)
