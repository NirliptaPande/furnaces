from PyALE import ale
import numpy as np
import joblib
import matplotlib.pyplot as plt
def plot_ale(idx):
    fig,ax = plt.subplots(1, figsize=(15, 7))
    ax.set_xlabel('ALE-%s'%name[idx])
    ale_eff = ale(X=x_train,model  = model,feature = echoes[:,idx],grid_size=50,include_CI=False,feature_type='continuous',fig = fig,ax = ax)

x_train = np.load('../data/train_zeros.npy')  
echoes = x_train[:,[6,14,16,19,27,34,35,37]].copy()
name = ['pftCrop','GDP','Distance_to_populated_area','WDPA_fracCover','road_density','Livestock','pop_density']
x_train = np.delete(x_train,[0,1,2,15,18,19,21,22,23,24,25,26,32,36],axis = 1)
model = joblib.load('../output/zeros/feature_all.joblib')
for i in range(echoes.shape[1]):
    plot_ale(i)

