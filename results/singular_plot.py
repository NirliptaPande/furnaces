import pandas as pd
from plot import cp_map
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
# plotting stuff
import matplotlib
matplotlib.use("Qt5Agg")


def plot(idx):

    y_eco = np.load('../output/zeros/y_pred_%s.npy' % (idx))
    y_mean = np.mean(np.reshape(y_eco, (-1, 57)), axis=1)
    df = pd.DataFrame(np.append(space_zero, y_mean.reshape(-1, 1),
                      axis=1), columns=['lat', 'lon', 'ign'])
    df = df.set_index(['lat', 'lon'])[['ign']]
    f, imax, im = cp_map(df, col='ign', cbrange=(0, 10), offset=(0, 0),
                         title='Ignitions %s- Zeros' % name[idx],
                         cmap=plt.get_cmap('nipy_spectral_r'), llc=(-180, -60),
                         projection=ccrs.PlateCarree(),
                         cb_kwargs={'cb_label': 'ignitions', 'cb_loc': 'right',
                                    'cb_extend': 'max'})
    f.savefig('../output/zeros/pred_%s.png' %
              var_name[idx], dpi=300, bbox_inches='tight')
    plt.close()

    y_eco = np.load('../output/month_mean/y_pred_%s.npy' % (idx))
    y_mean = np.mean(np.reshape(y_eco, (-1, 57)), axis=1)
    df = pd.DataFrame(np.append(space_custom, y_mean.reshape(-1, 1),
                      axis=1), columns=['lat', 'lon', 'ign'])
    df = df.set_index(['lat', 'lon'])[['ign']]
    f, imax, im = cp_map(df, col='ign', cbrange=(0, 10), offset=(0, 0),
                         title='Ignitions %s-Custom' % name[idx],
                         cmap=plt.get_cmap('nipy_spectral_r'), llc=(-180, -60),
                         projection=ccrs.PlateCarree(),
                         cb_kwargs={'cb_label': 'ignitions', 'cb_loc': 'right',
                                    'cb_extend': 'max'})
    f.savefig('../output/month_mean/pred_%s.png' %
              var_name[idx], dpi=300, bbox_inches='tight')
    plt.close()
# def plot_all(idx):
#     if (idx==0):
#         y_eco = np.load('../output/zeros/y_pred_all.npy'%(idx))
#     y_mean = np.mean(np.reshape(y_eco,(-1,57)),axis = 1)
#     df = pd.DataFrame(np.append(space,y_mean.reshape(-1,1),axis = 1),columns = ['lat','lon','ign'])
#     f, imax, im = cp_map(df, col='ign', cbrange=(0, 10), offset=(0, 0),
#                      title='Ignitions-Zeros-%s'%name[idx],
#                      cmap=plt.get_cmap('nipy_spectral_r'), llc=(-180, -60),
#                      projection=ccrs.PlateCarree(),
#                      cb_kwargs={'cb_label': 'ignitions', 'cb_loc': 'right',
#                                 'cb_extend': 'max'})
#     if(idx)f.savefig('../output/zeros/pred_%s.png'%name[idx], dpi=300, bbox_inches='tight')
#     plt.close()


name = ['CropLand', 'GDP', 'Distance to population',
        'Protected areas', 'Road Density', 'Livestock', 'Population Density']
var_name = ['pftCrop', 'GDP', 'Distance_to_populated_area',
            'WDPA_fracCover', 'road_density', 'Livestock', 'pop_density']
space_zero = np.load('../output/zeros/lat_lon.npy')
space_custom = np.load('../output/month_mean/lat_lon.npy')
for i in range(len(name)):
    plot(i)
# plot_all(0)
# plot_all(1)
