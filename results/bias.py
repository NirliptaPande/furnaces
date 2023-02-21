# This will calculate the bias, model will socio-economic facot-model without and plot it?
import pandas as pd
from plot import cp_map
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
# plotting stuff
import matplotlib
matplotlib.use("Qt5Agg")


def bias_zeros(idx):
    y_eco = np.load('../output/zeros/y_pred_%s.npy' % (idx))
    y_mean = np.mean(np.reshape(y_eco, (-1, 57)), axis=1)
    bias = y_mean - baseline
    err = y_mean - y_test_mean
    df = pd.DataFrame(np.append(space, bias.reshape(-1, 1),
                      axis=1), columns=['lat', 'lon', 'Bias'])
    df = df.set_index(['lat', 'lon'])[['Bias']]
    # print(df['Bias'])
    f, imax, im = cp_map(df, col='Bias', cbrange=(0, 10), offset=(0, 0),
                         title='Bias-%s' % name[idx],
                         cmap=plt.get_cmap('nipy_spectral_r'), llc=(-180, -60),
                         projection=ccrs.PlateCarree(),
                         cb_kwargs={'cb_label': 'ignitions', 'cb_loc': 'right',
                                    'cb_extend': 'max'})
    f.savefig('../output/zeros/bias_%s.png' %
              name[idx], dpi=300, bbox_inches='tight')
    plt.close()

    df = pd.DataFrame(np.append(space, err.reshape(-1, 1),
                      axis=1), columns=['lat', 'lon', 'Error'])
    df = df.set_index(['lat', 'lon'])[['Error']]
    f, imax, im = cp_map(df, col='Error', cbrange=(0, 10), offset=(0, 0),
                         title='Error-%s' % name[idx],
                         cmap=plt.get_cmap('nipy_spectral_r'), llc=(-180, -60),
                         projection=ccrs.PlateCarree(),
                         cb_kwargs={'cb_label': 'ignitions', 'cb_loc': 'right',
                                    'cb_extend': 'max'})
    f.savefig('../output/zeros/error_%s.png' %
              name[idx], dpi=300, bbox_inches='tight')
    plt.close()


def bias_mm(idx):
    y_eco = np.load('../output/month_mean/y_pred_%s.npy' % (idx))
    y_mean = np.mean(np.reshape(y_eco, (-1, 57)), axis=1)
    bias = y_mean - baseline
    err = y_mean - y_test_mean
    df = pd.DataFrame(np.append(space, bias.reshape(-1, 1),
                      axis=1), columns=['lat', 'lon', 'Bias'])
    df = df.set_index(['lat', 'lon'])[['Bias']]
    f, imax, im = cp_map(df, col='Bias', cbrange=(0, 10), offset=(0, 0),
                         title='Bias-%s' % name[idx],
                         cmap=plt.get_cmap('nipy_spectral_r'), llc=(-180, -60),
                         projection=ccrs.PlateCarree(),
                         cb_kwargs={'cb_label': 'number of ignitions', 'cb_loc': 'right',
                                    'cb_extend': 'max'})
    f.savefig('../output/month_mean/bias_%s.png' %
              name[idx], dpi=300, bbox_inches='tight')
    plt.close()
    df = pd.DataFrame(np.append(space, err.reshape(-1, 1),
                      axis=1), columns=['lat', 'lon', 'Error'])
    df = df.set_index(['lat', 'lon'])[['Error']]
    f, imax, im = cp_map(df, col='Error', cbrange=(0, 10), offset=(0, 0),
                         title='Error-%s' % name[idx],
                         cmap=plt.get_cmap('nipy_spectral_r'), llc=(-180, -60),
                         projection=ccrs.PlateCarree(),
                         cb_kwargs={'cb_label': 'number of ignitions', 'cb_loc': 'right',
                                    'cb_extend': 'max'})
    f.savefig('../output/month_mean/error_%s.png' %
              name[idx], dpi=300, bbox_inches='tight')
    plt.close()


baseline = np.load('../output/month_mean/y_pred_none.npy')
baseline = np.mean(np.reshape(baseline, (-1, 57)), axis=1)
y_test = np.load('../output/month_mean/y_test.npy')
y_test_mean = np.mean(np.reshape(y_test, (-1, 57)), axis=1)
space = np.load('../output/month_mean/lat_lon.npy')
name = ['pftCrop', 'GDP', 'Distance_to_populated_area',
        'WDPA_fracCover', 'road_density', 'Livestock', 'pop_density']
for i in range(len(name)):
    bias_mm(i)
baseline = np.load('../output/zeros/y_pred_none.npy')
y_test = np.load('../output/zeros/y_test.npy')
space = np.load('../output/zeros/lat_lon.npy')
y_test_mean = np.mean(np.reshape(y_test, (-1, 57)), axis=1)
baseline = np.mean(np.reshape(baseline, (-1, 57)), axis=1)
for i in range(len(name)):
    bias_zeros(i)
