# This code drops those columns where more than 1/3 data was missing
import numpy as np
train = np.load("../../data/new/train_new.npy")
train[np.where(train == -9999.0)] = np.NaN
train[np.isnan(train[:, 15]), 15] = 0
nan_mask = np.isnan(train)
nan_count = np.sum(nan_mask, axis=1)
df = np.load("../../data/new/custom_mean.npy")
#df.shape = (26378244, 30)
filter_mean = df[nan_count < 6]
#filter_mean.shape = (22482728, 30)
np.save("../../data/new/dropped_mean.npy", filter_mean)
df = np.load("../../data/new/custom_median.npy")
filter_median = df[nan_count < 6]
np.save("../../data/new/dropped_median.npy", filter_median)
