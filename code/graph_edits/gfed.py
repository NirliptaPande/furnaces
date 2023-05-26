import numpy as np

arr = np.load('../../data/custom_median.npy')
gfed = np.load('../../data/gfed_regions.npy',allow_pickle=True)

shaf = np.where(gfed==9)
seas = np.where(gfed==12)
boas = np.where(gfed==10)
aust = np.where(gfed==14)
euro = np.where(gfed==6)
lat = np.arange(90,-90,-0.25)
lon = np.arange(-180, 180, 0.25)

def data(cluster,name):
    cluster_lat = lat[cluster[0]]
    cluster_lon = lon[cluster[1]]
    arr_cluster = np.copy(arr[arr[:, 0] >= cluster_lat.min()])
    arr_cluster = arr_cluster[arr_cluster[:, 0] <= cluster_lat.max()]
    arr_cluster = arr_cluster[arr_cluster[:, 1] >= cluster_lon.min()]
    arr_cluster = arr_cluster[arr_cluster[:, 1] <= cluster_lon.max()]
    arr_cluster = arr_cluster[:,[0,1,6,8,14,15,16,22,25,26,28]]
    #lat, lon, pftCrop, pftShrubBD, GDP, ign, faPAR, precip, pop_density, distance to population, temp_max 
    dim = np.vstack((cluster_lat,cluster_lon))
    cluster_val = []
    for i in range(dim.shape[1]):
        mask = np.all(arr_cluster[:, :2] == dim[:,i], axis=1)
        if np.any(mask)==True:
            cluster_val.append(arr_cluster[mask])  
    np.save('../../data/%s.npy'%name,cluster_val)

data(shaf,'shaf')
data(seas,'seas')
data(boas,'boas')
data(aust,'aust')
data(euro,'euro')
