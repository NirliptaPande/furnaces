import numpy as np
import gc

train_month_mean = np.load('../../data/new/custom_mean.npy')
train_month_mean = train_month_mean.reshape(
    (-1, 9, 12, train_month_mean.shape[-1]))
annual_mean = np.mean(train_month_mean, axis=2)
annual_mean = np.reshape(annual_mean, (-1, annual_mean.shape[-1]))
np.save("../../data/new/annual_mean.npy", annual_mean)
del (train_month_mean)
del (annual_mean)
gc.collect()

month_median = np.load('../../data/new/custom_median.npy')
month_median = month_median.reshape((-1, 9, 12, month_median.shape[-1]))
annual_median = np.mean(month_median, axis=2)
annual_median = np.reshape(annual_median, (-1, annual_median.shape[-1]))
np.save("../../data/new/annual_median.npy", annual_median)
