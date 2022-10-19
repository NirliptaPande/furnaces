"""
This script will balance the dataset according to the ignitions, and save the lat, lon inside data
"""
import numpy as np
import gc
train_x = np.load("../data/train_zeros.npy")

"""
Balancing the dataset because RF is really sensitive to it
"""
no_ign = np.where(train_x[:,15]==0)
ign = np.nonzero(train_x[:,15])

new_no_ign = train_x[np.random.choice(no_ign[0],ign[0].shape,replace = False)]
new_ign = train_x[np.random.choice(ign[0],ign[0].shape[0],replace = False)]
x_train = np.concatenate((new_ign,new_no_ign),axis = 0)
np.random.shuffle(x_train)
np.save('../data/zeros_balanced.npy',x_train)
#idx = np.copy(x_train[:,[0,1]])
del(train_x)
del(x_train)
gc.collect()

train_x = np.load("../data/train_monthmean.npy")
train_x = np.reshape(train_x,(-1,train_x.shape[-1]))
"""
Balancing the dataset because RF is really sensitive to it
"""
no_ign = np.where(train_x[:,15]==0)
ign = np.nonzero(train_x[:,15])

new_no_ign = train_x[np.random.choice(no_ign[0],ign[0].shape,replace = False)]
new_ign = train_x[np.random.choice(ign[0],ign[0].shape[0],replace = False)]
x_train = np.concatenate((new_ign,new_no_ign),axis = 0)
np.random.shuffle(x_train)
np.save('../data/mm_balanced.npy',x_train)
#idx = np.copy(x_train[:,[0,1]])