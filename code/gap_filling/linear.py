#In this file, we will use linear interpolation across a month to fill in the missing data, where needed, or we can fill in zeros if needed like in ignition
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import gc
train = np.load('../../data/train_sans_ocean.npy')
pixels = train.shape[0]//108
spatial = np.reshape(train,(pixels,9,12,train.shape[1]))#converts the array into 3D, first is all pixels, second is time steps and thrid is features
