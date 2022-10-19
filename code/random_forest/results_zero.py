import joblib
import numpy as np
#def plot_pred(eco):
#def predict(idx):
test_x = np.load('../../data/test_zeros.npy')
test_y = np.copy(test_x[:,15])
echoes = test_x[:,[6,14,16,27,34,35,37]].copy()
#Livestock 34, road_density 35,pop_density 37,Distance_to_populated_area 16,GDP 14,pftCrop 6
test_x = np.delete(test_x,[0,1,2,6,14,15,16,18,19,21,22,23,24,25,26,27,32,34,35,36,37],axis = 1)