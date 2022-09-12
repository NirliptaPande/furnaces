from unicodedata import name
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool
def graphs(idx):
    names =  ['agb', 'pft_fracCover', 'sm', 'pftCrop',
       'pftHerb', 'pftShrubBD', 'pftShrubNE', 'pftTreeBD', 'pftTreeBE',
       'pftTreeND', 'pftTreeNE', 'GDP', 'ign', 'Distance_to_populated_areas',
       'fPAR', 'LAI', 'NLDI', 'vod_K_anomalies', 'FPAR_12mon', 'LAI_12mon',
       'Vod_k_anomaly_12mon', 'FPAR_06mon', 'LAI_06mon', 'Vod_k_anomaly_06mon',
       'WDPA_fracCover', 'dtr', 'pet', 'tmx', 'wet', 'Biome', 'precip',
       'Livestock', 'road_density', 'topo', 'pop_density']
    name = names[idx]
    col = train[:,idx]
    plt.hist(col,bins = np.arange(np.amin(col),np.amax(col),(np.amax(col)-np.amin(col))/100))
    plt.xlabel('%s'%name)
    plt.ylabel('Freq')
    plt.title("Frequency of %s"%name)
    plt.savefig('../output/%s_with_zeros.png'%name)
    plt.close()
    col = col[col!=0]
    plt.hist(col,bins = np.arange(np.amin(col),np.amax(col),(np.amax(col)-np.amin(col))/100))
    plt.xlabel('%s'%name)
    plt.ylabel('Freq')
    plt.title("Frequency of %s without zeros"%name)
    plt.savefig('../output/%s_sans_zeros.png'%name)
    plt.close()
    col_ln = np.log(col)
    plt.hist(col_ln,bins = np.arange(np.amin(col_ln),np.amax(col_ln),np.log((np.amax(col)-np.amin(col))/100)))
    plt.xlabel('Log_e %s'%name)
    plt.ylabel('Freq')
    plt.title("Frequency of %s log_e"%name)
    plt.savefig('../output/%s_log_e.png'%name)
if __name__ == '__main__':
    global train
    train = np.load('../data/train_sans_ocean.npy')
    np.nan_to_num(train,copy=False)
    np.save('../data/train_sans_ocean.npy',train)
    train = np.delete(train,[0,1,2],1)
    idxs = np.arange(train.shape[1])
    with Pool() as p:
        p.map(graphs,idxs)
    p.close()
    p.join()
