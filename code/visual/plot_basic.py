from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
import gc
global train
def plot_one(idx):
    names = ['agb', 'pft_fracCover', 'sm', 'pftCrop',
       'pftHerb', 'pftShrubBD', 'pftShrubNE', 'pftTreeBD', 'pftTreeBE',
       'pftTreeND', 'pftTreeNE', 'GDP', 'ign', 'Distance_to_populated_areas',
       'fPAR', 'LAI', 'NLDI', 'vod_K_anomalies', 'FPAR_12mon', 'LAI_12mon',
       'Vod_k_anomaly_12mon', 'FPAR_06mon', 'LAI_06mon', 'Vod_k_anomaly_06mon',
       'WDPA_fracCover', 'dtr', 'pet', 'tmx', 'wet', 'Biome', 'precip',
       'Livestock', 'road_density', 'topo', 'pop_density']
    var = train[:,idx]
    name = names[idx]
    plt.hist(var,bins = np.arange(np.nanmin(var),np.nanmax(var),((np.nanmax(var)-np.nanmin(var))/100)))
    plt.xlabel('%s'%name)
    plt.ylabel('Freq')
    plt.title("Distribution of %s with NaN"%name)
    plt.savefig('../output/%s_with_nan'%name)
    plt.close()
    #Hypothesis: Shouldn't make a difference, with or without nan
    var_nan = var[~np.isnan(var)]
    plt.hist(var_nan,bins = np.arange(np.amin(var_nan),((np.amax(var_nan)-np.amin(var_nan))/100)))
    plt.xlabel('%s'%name)
    plt.ylabel('Freq')
    plt.title("Distribution of %s without NaN"%name)
    plt.savefig('../output/%s_without_nan'%name)
    plt.close()
    del(var_nan)
    gc.collect()
    #Putting in zeros should zoom out the graph
    var_zero = np.nan_to_num(var,nan =0)
    plt.hist(var_zero,bins = np.arange(np.amin(var_zero),((np.amax(var_zero)-np.amin(var_zero))/100)))
    plt.xlabel('%s'%name)
    plt.ylabel('Freq')
    plt.title("Distribution of %s with Zero"%name)
    plt.savefig('../output/%s_with_zero'%name)
    plt.close()
    del(var_zero)
    gc.collect()
    var_mean = np.nan_to_num(var,nan =np.nanmean(var))
    x,_,_ = plt.hist(var_mean,bins = np.arange(np.amin(var_mean),((np.amax(var_mean)-np.amin(var_mean))/100)))
    plt.xlabel('%s'%name)
    plt.ylabel('Freq')
    plt.vlines(var_mean,x.min(),x.max(),colors='r',linestyle= 'dashed', label = 'mean')
    plt.legend()
    plt.title("Distribution of %s with Zero"%name)
    plt.savefig('../output/%s_with_zero'%name)
    plt.close()
    del(var_mean)
    gc.collect()
    var_median = np.nan_to_num(var,nan =np.nanmedian(var))
    x,_,_ = plt.hist(var_median,bins = np.arange(np.amin(var_median),((np.amax(var_median)-np.amin(var_median))/100)))
    plt.xlabel('%s'%name)
    plt.ylabel('Freq')
    plt.title("Distribution of %s with Zero"%name)
    plt.vlines(var_median,x.min(),x.max(),colors='r',linestyle= 'dashed', label = 'median')
    plt.legend()
    plt.savefig('../output/%s_with_zero'%name)
    plt.close()
    del(var_median)
    gc.collect()

if __name__ == '__main__':
    train = np.load('../data/train_sans_ocean.npy')
    train = np.delete(train,[0,1,2],1)
    idxs = np.arange(train.shape[1])
    with Pool() as p:
        p.map(plot_one,idxs)
    p.close()
    p.join()