#Compiles data using split npy files from split.py and fills in median values because values are skewed
#Note: If normally distributed data, use mean, else median
import numpy as np
#from multiprocessing import Pool
train_num = 108
test_num = 57
land = np.load('../../data/land_coords.npy')
indices = np.arange(72)
indices_arr = np.arange(14400)
arr = np.load('../../data/new/train_0.npy')
temp = np.zeros((land.shape[0]*train_num,arr.shape[1]))
c = 0
del(arr)
for index in indices:
    arr = np.load('../../data/new/train_%i.npy'%index)
    for i in indices_arr:
        lat = arr[i*train_num][0]
        lon = arr[i*train_num][1]
        if np.any(np.logical_and((lat==land[:,0]),(lon==land[:,1])))==True:
            temp[c*train_num:(c+1)*train_num] = arr[i*train_num:(i+1)*train_num]
            c+=1
# col_median = np.nanmedian(temp,axis = 0)
# col_median = np.reshape(col_median,((-1,1)))
# idx = np.where(np.isnan(temp))
# temp[idx] = np.take(col_median,idx[1])
# np.save('../data/new/train_median.npy',temp)
np.save('../../data/new/train_new.npy',temp)#running this again because ign doesn't make sense at all
#But I don't have the data here
