#This file will participate in grid serach for values filled with zeros
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
x_train=  np.load("/home/npande/Desktop/furnaces/data/train_zeros.npy")
y_train = x_train[:,15].copy()
x_train = np.delete(x_train,[14,15,34,35,37],axis = 1)
    #0: 34 Livestock
    #1: 35 road_density
    #2: 37 pop_density
    #3: 14 GDP
    #4: 15 Ignition
x_test=  np.load("/home/npande/Desktop/furnaces/data/test_zeros.npy")
y_test = x_test[:,15].copy()
x_test = np.delete(x_test,[14,15,34,35,37],axis = 1)
    #0: 34 Livestock
    #1: 35 road_density
    #2: 37 pop_density
    #3: 14 GDP
    #4: 15 Ignition
"""['lat', 'lon', 'time', 'agb', 'pft_fracCover', 'sm', 'pftCrop',
       'pftHerb', 'pftShrubBD', 'pftShrubNE', 'pftTreeBD', 'pftTreeBE',
       'pftTreeND', 'pftTreeNE', 'Distance_to_populated_areas',
       'fPAR', 'LAI', 'NLDI', 'vod_K_anomalies', 'FPAR_12mon', 'LAI_12mon',
       'Vod_k_anomaly_12mon', 'FPAR_06mon', 'LAI_06mon', 'Vod_k_anomaly_06mon',
       'WDPA_fracCover', 'dtr', 'pet', 'tmx', 'wet', 'Biome', 'precip', 'topo']"""
model = RandomForestRegressor()
grid_params  = {
    'bootstrap' : [True,False],
    'n_estimators' : [50,100,250,400,600,850,1200,1500],
    'criterion' : ['gini', 'entropy', 'log_loss'],
    'min_samples_leaf' : [1,2,3,5],
    'max_features' : [5,7,10],
    'warm_start': [True, False],
    'min_impurity_decrease': [0.0, 0.2, 0.4, 0.6, 0.8],
    'min_samples_split': [2, 3, 4, 5, 6, 7, 8]
}
random_search = RandomizedSearchCV(
    estimator = model, 
    param_grids = grid_params,
    cv = 5,
    n_jobs = -1,
    random_state = 42,
    n_iter=1000, 
    scoring='accuracy')
#n_estimators=100,
#  criterion='gini',
#  max_depth=None,
#  min_samples_split=2, 
# min_samples_leaf=1, 
# min_weight_fraction_leaf=0.0,
#  max_features='sqrt',
#  max_leaf_nodes=None,
#  min_impurity_decrease=0.0,
#  bootstrap=True,
#  oob_score=False,
#  n_jobs=None,
#  random_state=None, 
# verbose=0,
#  warm_start=False, 
# class_weight=None,
#  ccp_alpha=0.0,
#  max_samples=None)
