#This code fills in data with zeros to test it against ones with linear interpolation across time
import numpy as np
train =  np.load("/home/npande/Desktop/furnaces/data/train_sans_ocean.npy")
np.nan_to_num(train, copy= False,nan = 0)
np.save("/home/npande/Desktop/furnaces/data/train_zeros.npy",train)
test = np.load("/home/npande/Desktop/furnaces/data/test_sans_ocean.npy")
np.nan_to_num(test, copy= False,nan = 0)
np.save("/home/npande/Desktop/furnaces/data/test_zeros.npy",test)