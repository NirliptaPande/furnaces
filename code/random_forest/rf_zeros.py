import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
def rfreg(eco):
    xtrain = np.append(x_train,np.transpose([echoes[:,eco]]),axis = 1)
    model = RandomForestRegressor(n_estimators=500, criterion='squared_error', random_state=42,
                                     n_jobs=60, min_samples_split= 5,bootstrap=True,
                                   max_features = 7,oob_score=True)
    # Note: I am aware that max_features shows an error, but it's not, it runs perfectly fine, it is supposed to accept int
    model.fit(xtrain,train_y)
    joblib.dump(model, "../../output/zeros/feature%s.joblib"%str(eco),compress = True)

x_train = np.load("../../data/zeros_balanced.npy")
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
echoes = x_train[:,[6,14,16,27,34,35,37]].copy()
#Livestock 34, road_density 35,pop_density 37,Distance_to_populated_area 16,GDP 14,pftCrop 6
x_train = np.delete(x_train,[0,1,2,6,14,15,16,18,19,21,22,23,24,25,26,27,32,34,35,36,37],axis = 1)
"""
topo 36
Biome 32
lon 1
lat 0
time 2
LAI 18
FPAR_12mon 21
LAI_12mon 22
Vod_k_anomaly_12mon 23
FPAR_06mon 24
LAI_06mon 25
Vod_k_anomaly_06mon 26
pft_fracCover 4
tmx 30
dtr 28
NLDI 19
"""
"""
Now taking out the 'ign' and copying and removing the socio-economic variables
"""
for i in range(echoes.shape[1]):
    rfreg(i)