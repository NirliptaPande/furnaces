import numpy as np
from multiprocessing import Pool
from netCDF4 import Dataset
if __name__ == '__main__':
    df1 = Dataset('../data/grid.nc')
    mask = df1['subset_flag'][:]
    lat = df1['lat'][:]
    lon = df1['lon'][:]
    mask = np.array(mask)
    lat = np.array(lat)
    lon = np.array(lon)
    lat = lat +.125
    lon = lon + .125
    land = np.where(mask==1)
    indices = np.arange(72)
    