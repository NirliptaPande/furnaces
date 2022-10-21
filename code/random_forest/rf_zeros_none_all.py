#This runs random forest with zeros in the data with none and all the features
import gc
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
x_train = np.load("../../data/zeros_balanced.npy")
train_y = x_train[:,15].copy()#taking out the ign values
#echoes = x_train[:,[6,14,16,27,34,35,37]].copy()
"""
['lat 0', 'lon 1', 'time 2', 'agb 3', 'pft_fracCover 4', 'sm 5', 'pftCrop 6',
       'pftHerb 7', 'pftShrubBD 8', 'pftShrubNE 9', 'pftTreeBD 10', 'pftTreeBE 11',
       'pftTreeND 12', 'pftTreeNE 13', 'GDP 14', 'ign 15', 'Distance_to_populated_areas 16',
       'fPAR 17', 'LAI 18', 'NLDI 19', 'vod_K_anomalies 20', 'FPAR_12mon 21', 'LAI_12mon 22',
       'Vod_k_anomaly_12mon 23', 'FPAR_06mon 24', 'LAI_06mon 25', 'Vod_k_anomaly_06mon 26',
       'WDPA_fracCover 27', 'dtr 28', 'pet 29', 'tmx 30', 'wet 31', 'Biome 32', 'precip 33',
       'Livestock 34', 'road_density 35', 'topo 36', 'pop_density 37']
"""
x_train = np.delete(x_train,[0,1,2,15,18,19,21,22,23,24,25,26,32,36],axis = 1)#Train with the economic factors
model = RandomForestRegressor(n_estimators=500, criterion='squared_error', random_state=42,
                                    n_jobs=60, min_samples_split= 5,bootstrap=True,
                                max_features = 7, oob_score=True)
# Note: I am aware that max_features shows an error, but it's not, it runs perfectly fine, it is supposed to accept int
model.fit(x_train,train_y)
joblib.dump(model, "../../output/zeros/feature_all.joblib",compress = True)
"""
After deletion of variables in line 45, the x_train looks like
['agb 0', 'pft_fracCover 1', 'sm 2', 'pftCrop 3',
       'pftHerb 4', 'pftShrubBD 5', 'pftShrubNE 6', 'pftTreeBD 7', 'pftTreeBE 8',
       'pftTreeND 9', 'pftTreeNE 10', 'GDP 11', 'Distance_to_populated_areas 12',
       'fPAR 13', 'vod_K_anomalies 14', 'WDPA_fracCover 15', 'dtr 16', 'pet 17', 'tmx 18', 'wet 19', 'precip 20',
       'Livestock 21', 'road_density 22', 'pop_density 23']
and we wish to remove, pftCrop 3, GDP 11, Distance_to_populated_areas 12, WDPA_fracCover15, livestock 21, road_density 22, pop_density 23
"""
x_train = np.delete(x_train,[3,11,12,15,21,22,23],axis = 1)
model = RandomForestRegressor(n_estimators=500, criterion='squared_error', random_state=42,
                                    n_jobs=60, min_samples_split= 5,bootstrap=True,
                                max_features = 7,oob_score=True)
# Note: I am aware that max_features shows an error, but it's not, it runs perfectly fine, it is supposed to accept int
model.fit(x_train,train_y)
joblib.dump(model, "../../output/zeros/feature_none.joblib",compress = True)
