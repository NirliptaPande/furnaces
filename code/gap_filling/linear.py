#In this file, we will use linear interpolation across a month to fill in the missing data, where needed, or we can fill in zeros if needed like in ignition
import numpy as np
#from mpl_toolkits.basemap import Basemap
#import matplotlib.pyplot as plt
import gc
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
