#This file will participate in grid serach for values filled with zeros
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import joblib
#x_train=  np.load("/home/npande/Desktop/furnaces/data/train_zeros.npy")
x_train=  np.load("/home/npande/furnaces/data/train_zeros.npy")
y_train = x_train[:,15].copy()
x_train = np.delete(x_train,[14,15,34,35,37],axis = 1)
"""['lat', 'lon', 'time', 'agb', 'pft_fracCover', 'sm', 'pftCrop',
       'pftHerb', 'pftShrubBD', 'pftShrubNE', 'pftTreeBD', 'pftTreeBE',
       'pftTreeND', 'pftTreeNE', 'Distance_to_populated_areas',
       'fPAR', 'LAI', 'NLDI', 'vod_K_anomalies', 'FPAR_12mon', 'LAI_12mon',
       'Vod_k_anomaly_12mon', 'FPAR_06mon', 'LAI_06mon', 'Vod_k_anomaly_06mon',
       'WDPA_fracCover', 'dtr', 'pet', 'tmx', 'wet', 'Biome', 'precip', 'topo']"""
model = RandomForestRegressor()
grid_params  = {
    'bootstrap' : [True,False],#whether to sample with replacement or not 
    'n_estimators' : [50,100,250,400,600,850,1200,1500],# number of trees
    'criterion' : ['squared_error', 'absolute_error', 'poisson'],#To decerease the impurity
    'min_samples_leaf' : [1,3,5,7,10],
    # A split point at any depth will only be considered if 
    # it leaves at least min_samples_leaf training samples in both branches. 
    #This may have the effect of smoothing the model, especially in regression.
    'max_features' : [5,7,10], #To split a node, how many features to consider
    'warm_start': [True, False],
    'min_impurity_decrease': [0.0, 0.03, 0.07, 0.1, 0.13],#A node will split only if impurity decreases by at least
    'min_samples_split': [2, 3, 4, 5, 6, 7, 8]#You should have 
    #at least these many samples to split a node
}
random_search = RandomizedSearchCV(
    estimator = model, 
    param_distributions = grid_params,
    cv = 5,
    n_jobs = 8,
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
random_search.fit(x_train,y_train)
print(random_search.best_params_)
joblib.dump(random_search,'../../output/zero_random.pkl')