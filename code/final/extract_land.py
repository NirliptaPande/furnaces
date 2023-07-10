#This file extracts land coords from a netcdf file
from netCDF4 import Dataset
import numpy as np
rf = Dataset("../../data/grid.nc")
mask = rf['subset_flag'][:]
lat = rf['lat'][:]
lon = rf['lon'][:]
mask = np.array(mask)
lat = np.array(lat)
lon = np.array(lon)
lat = lat +.125
lon = lon + .125
land = np.where(mask==1)
land_coords = np.zeros((len(land[0]),2))
lat = []
lon = []
for i in range(len(land[0])):
    coord = land[0][i]
    lat.append((((coord//720)-360)/4))
    lon.append((((coord%720)-720)/4))
coords = list(zip(lat,lon))
land_coords = np.array(coords)
np.save("../../data/land_coords.npy",land_coords)
#You can now use this land coords to remove oceans from the dataset