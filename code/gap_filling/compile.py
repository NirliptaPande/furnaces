#Compiles data using split npy files from split.py and fills in median values because values are skewed
#Note: If normally distributed data, use mean, else median
import numpy as np
#from multiprocessing import Pool
land = np.load('../data/land_coords.npy')
indices = np.arange(72)
indices_arr = np.arange(14400)
arr = np.load('../data/train_0.npy')
temp = np.zeros((land.shape[0]*108,arr.shape[1]))
c = 0
del(arr)
for index in indices:
    arr = np.load('../data/train_%i.npy'%index)
    for i in indices_arr:
        lat = arr[i*108][0]
        lon = arr[i*108][1]
        if np.any(np.logical_and((lat==land[:,0]),(lon==land[:,1])))==True:
            temp[c*108:(c+1)*108] = arr[i*108:(i+1)*108]
            c+=1
col_median = np.nanmedian(temp,axis = 0)
col_median = np.reshape(col_median,((-1,1)))
idx = np.where(np.isnan(temp))
temp[idx] = np.take(col_median,idx[1])
np.save('../data/train_median.npy',temp)