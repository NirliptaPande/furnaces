"""
mm stands for monthly mean. 
This script just replaces the file path of train_x in rf_zeros.py to the output from ../gap_filling/mean_loop.py which is stored in ../../data as train_monthmean.npy
"""
import gc
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
def rfreg(eco):
    xtrain = np.append(x_train,np.transpose([echoes[:,eco]]),axis = 1)
    model = RandomForestRegressor(n_estimators=500, criterion='squared_error', random_state=42,
                                     n_jobs=60, min_samples_split= 5,bootstrap=True,
                                   max_features = 7,oob_score=True)# Note: I am aware that max_features shows an error, but it's not, it runs perfectly fine
    model.fit(xtrain,train_y)
    joblib.dump(model, "../../output/month_mean/feature%s.joblib"%str(eco))

train_x = np.load("../../data/train_monthmean.npy")

"""
Balancing the dataset because RF is really sensitive to it
"""
no_ign = np.where(train_x[:,15]==0)
ign = np.nonzero(train_x[:,15])

#ign_train  = train_x[ign]
#no_ign_train = train_x[no_ign]
"""I have absolutely no idea why ign is larger than no ign and if I should run the model or not"""

new_no_ign = train_x[np.random.choice(no_ign[0],ign[0].shape,replace = False)]
new_ign = train_x[np.random.choice(ign[0],ign[0].shape[0],replace = False)]
x_train = np.concatenate((new_ign,new_no_ign),axis = 0)
np.random.shuffle(x_train)
"""
Balancing and shuffling of the dataset is done
"""
train_y = x_train[:,15].copy()#taking out the ign values
"""
['lat 0', 'lon 1', 'time 2', 'agb 3', 'pft_fracCover 4', 'sm 5', 'pftCrop 6',
       'pftHerb 7', 'pftShrubBD 8', 'pftShrubNE 9', 'pftTreeBD 10', 'pftTreeBE 11',
       'pftTreeND 12', 'pftTreeNE 13', 'GDP 14', 'ign 15', 'Distance_to_populated_areas 16',
       'fPAR 17', 'LAI 18', 'NLDI 19', 'vod_K_anomalies 20', 'FPAR_12mon 21', 'LAI_12mon 22',
       'Vod_k_anomaly_12mon 23', 'FPAR_06mon 24', 'LAI_06mon 25', 'Vod_k_anomaly_06mon 26',
       'WDPA_fracCover 27', 'dtr 28', 'pet 29', 'tmx 30', 'wet 31', 'Biome 32', 'precip 33',
       'Livestock 34', 'road_density 35', 'topo 36', 'pop_density 37']
"""
echoes = x_train[:,[34,35,37,14]].copy()
x_train = np.delete(x_train,[2,14,15,18,19,21,22,23,24,25,26,34,35,36,37],axis = 1)
"""
Now taking out the 'ign' and copying and removing the socio-economic variables
"""
del(train_x)
gc.collect()
for i in range(echoes.shape[1]):
    rfreg(i)