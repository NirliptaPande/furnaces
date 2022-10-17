#This runs random forest with zeros in the data with none and all the features
import gc
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
train_x = np.load("../../data/train_monthmean.npy")
train_x = np.reshape(train_x,(-1,train_x.shape[-1]))
"""
Balancing the dataset because RF is really sensitive to it
"""
no_ign = np.where(train_x[:,15]==0)
ign = np.nonzero(train_x[:,15])

#ign_train  = train_x[ign]
#no_ign_train = train_x[no_ign]
new_no_ign = train_x[np.random.choice(no_ign[0],ign[0].shape,replace = False)]
new_ign = train_x[np.random.choice(ign[0],ign[0].shape[0],replace = False)]
x_train = np.concatenate((new_ign,new_no_ign),axis = 0)
np.random.shuffle(x_train)
"""
Balancing and shuffling of the dataset is done
"""
train_y = x_train[:,15].copy()#taking out the ign values
#echoes = x_train[:,[34,35,37,14]].copy()
"""
['lat 0', 'lon 1', 'time 2', 'agb 3', 'pft_fracCover 4', 'sm 5', 'pftCrop 6',
       'pftHerb 7', 'pftShrubBD 8', 'pftShrubNE 9', 'pftTreeBD 10', 'pftTreeBE 11',
       'pftTreeND 12', 'pftTreeNE 13', 'GDP 14', 'ign 15', 'Distance_to_populated_areas 16',
       'fPAR 17', 'LAI 18', 'NLDI 19', 'vod_K_anomalies 20', 'FPAR_12mon 21', 'LAI_12mon 22',
       'Vod_k_anomaly_12mon 23', 'FPAR_06mon 24', 'LAI_06mon 25', 'Vod_k_anomaly_06mon 26',
       'WDPA_fracCover 27', 'dtr 28', 'pet 29', 'tmx 30', 'wet 31', 'Biome 32', 'precip 33',
       'Livestock 34', 'road_density 35', 'topo 36', 'pop_density 37']
"""
x_train = np.delete(x_train,[2,15,18,19,21,22,23,24,25,26,36],axis = 1)#Train with the economic factors
del(train_x)
gc.collect()
model = RandomForestRegressor(n_estimators=500, criterion='squared_error', random_state=42,
                                    n_jobs=60, min_samples_split= 5,bootstrap=True,
                                max_features = 7, oob_score=True)
# Note: I am aware that max_features shows an error, but it's not, it runs perfectly fine, it is supposed to accept int
model.fit(x_train,train_y)
joblib.dump(model, "../../output/month_mean/feature_all.joblib")

x_train = np.delete(x_train,[13,24,25,26],axis = 1)
"""
After deletion of variables in line 45, the x_train looks like
['lat 0', 'lon 1', 'agb 2', 'pft_fracCover 3', 'sm 4', 'pftCrop 5',
       'pftHerb 6', 'pftShrubBD 7', 'pftShrubNE 8', 'pftTreeBD 9', 'pftTreeBE 10',
       'pftTreeND 11', 'pftTreeNE 12', 'GDP 13', 'Distance_to_populated_areas 14',
       'fPAR 15', 'vod_K_anomalies 16', 'WDPA_fracCover 17', 'dtr 18', 'pet 19', 'tmx 20', 'wet 21', 'Biome 22', 'precip 23',
       'Livestock 24', 'road_density 25', 'pop_density 26']
and we wish to remove, GDP 13, livestock 24, road_density 25, pop_density 26
"""
x_train = np.delete(x_train,[13,24,25,26],axis = 1)
model = RandomForestRegressor(n_estimators=500, criterion='squared_error', random_state=42,
                                    n_jobs=60, min_samples_split= 5,bootstrap=True,
                                max_features = 7,oob_score=True)
# Note: I am aware that max_features shows an error, but it's not, it runs perfectly fine, it is supposed to accept int
model.fit(x_train,train_y)
joblib.dump(model, "../../output/month_mean/feature_none.joblib")
