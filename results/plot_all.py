import pandas as pd
from plot import cp_map
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
# plotting stuff
import matplotlib
matplotlib.use("Qt5Agg")


space = np.load('../output/zeros/lat_lon.npy')
y_eco = np.load('../output/zeros/y_pred_all.npy')
y_mean = np.mean(np.reshape(y_eco, (-1, 57)), axis=1)
df = pd.DataFrame(np.append(space, y_mean.reshape(-1, 1),
                  axis=1), columns=['lat', 'lon', 'ign'])
df = df.set_index(['lat', 'lon'])[['ign']]
f, imax, im = cp_map(df, col='ign', cbrange=(0, 10), offset=(0, 0),
                     title='Ignitions-Zeros-all',
                     cmap=plt.get_cmap('nipy_spectral_r'), llc=(-180, -60),
                     projection=ccrs.PlateCarree(),
                     cb_kwargs={'cb_label': 'ignitions', 'cb_loc': 'right',
                                'cb_extend': 'max'})
f.savefig('../output/zeros/pred_all.png', dpi=300, bbox_inches='tight')
plt.close()


y_eco = np.load('../output/zeros/y_pred_none.npy')
y_mean = np.mean(np.reshape(y_eco, (-1, 57)), axis=1)
df = pd.DataFrame(np.append(space, y_mean.reshape(-1, 1),
                  axis=1), columns=['lat', 'lon', 'ign'])
df = df.set_index(['lat', 'lon'])[['ign']]
f, imax, im = cp_map(df, col='ign', cbrange=(0, 10), offset=(0, 0),
                     title='Ignitions-Zeros-all',
                     cmap=plt.get_cmap('nipy_spectral_r'), llc=(-180, -60),
                     projection=ccrs.PlateCarree(),
                     cb_kwargs={'cb_label': 'ignitions', 'cb_loc': 'right',
                                'cb_extend': 'max'})
f.savefig('../output/zeros/pred_none.png', dpi=300, bbox_inches='tight')
plt.close()


space = np.load('../output/month_mean/lat_lon.npy')
y_eco = np.load('../output/month_mean/y_pred_all.npy')
y_mean = np.mean(np.reshape(y_eco, (-1, 57)), axis=1)
df = pd.DataFrame(np.append(space, y_mean.reshape(-1, 1),
                  axis=1), columns=['lat', 'lon', 'ign'])
df = df.set_index(['lat', 'lon'])[['ign']]
f, imax, im = cp_map(df, col='ign', cbrange=(0, 10), offset=(0, 0),
                     title='Ignitions-Zeros-all',
                     cmap=plt.get_cmap('nipy_spectral_r'), llc=(-180, -60),
                     projection=ccrs.PlateCarree(),
                     cb_kwargs={'cb_label': 'ignitions', 'cb_loc': 'right',
                                'cb_extend': 'max'})
f.savefig('../output/month_mean/pred_all.png', dpi=300, bbox_inches='tight')
plt.close()


y_eco = np.load('../output/month_mean/y_pred_none.npy')
y_mean = np.mean(np.reshape(y_eco, (-1, 57)), axis=1)
df = pd.DataFrame(np.append(space, y_mean.reshape(-1, 1),
                  axis=1), columns=['lat', 'lon', 'ign'])
df = df.set_index(['lat', 'lon'])[['ign']]
f, imax, im = cp_map(df, col='ign', cbrange=(0, 10), offset=(0, 0),
                     title='Ignitions-monthmean-all',
                     cmap=plt.get_cmap('nipy_spectral_r'), llc=(-180, -60),
                     projection=ccrs.PlateCarree(),
                     cb_kwargs={'cb_label': 'ignitions', 'cb_loc': 'right',
                                'cb_extend': 'max'})
f.savefig('../output/month_mean/pred_none.png', dpi=300, bbox_inches='tight')
plt.close()
