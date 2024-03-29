import numpy as np
import joblib
# from mpl_toolkits.basemap import Basemap
# import matplotlib.pyplot as plt
# import gc
def plot_feature(idx):
    xtest = np.append(x_test,np.transpose([echoes[:,idx]]),axis = 1)
    model = joblib.load('../output/month_mean/feature%s.joblib'%idx)
    y_pred = model.predict(xtest)
    np.save('../output/month_mean/y_pred_%s.npy'%str(idx),y_pred)
    #err_mean[idx] = np.mean(((y_test_mean-np.mean(y_pred))**2))
    # y_pred_mean = np.mean(np.reshape(y_pred,(-1,57)),axis = 1)
    # err[idx] = np.divide(np.mean(((y_test-y_pred)**2)),np.mean(y_test**2))
    # err_mean[idx] = np.divide(np.mean(((y_test_mean-np.mean(y_pred))**2)),y_test_mean**2)
    # x = np.linspace(-89.5,90,720)
    # y = np.linspace(-179.5,180,1440)
    # lons,lats = np.meshgrid(y,x)
    # grid = np.zeros((720,1440))
    # y_pred_mean = np.reshape(y_pred_mean,(-1,1))
    # y_pred_mean = np.append(lon.reshape(-1,1),y_pred_mean,axis = 1)#lon
    # y_pred_mean = np.append(lat.reshape(-1,1),y_pred_mean,axis = 1)#lat
    # for i in range(y_pred_mean.shape[0]):
    #     grid[int(y_pred_mean[:,0][i]+89.5)*4][int(y_pred_mean[:,1][i]+179.5)*4] = y_pred_mean[:,2][i]
    # fig = plt.figure(figsize=(35,30), edgecolor='w')
    # map = Basemap(projection='cyl',resolution='c',llcrnrlat=-90,urcrnrlat=90,llcrnrlon=-180,urcrnrlon=180)
    # map.pcolormesh(lons,lats,grid,cmap = 'plasma')
    # map.drawcoastlines(color='lightgray')
    # map.drawparallels(np.arange(-90.,91.,15.),color='grey')
    # map.drawmeridians(np.arange(-180.,181.,30.),color='grey')
    # plt.title(" Predicted Ignitions - %s"%name[idx])
    # plt.colorbar()
    # plt.savefig('../output/month_mean/%s'%name[idx])
    # plt.close()
    # del(model)
    # gc.collect()
x_test  = np.load('../data/test_monthmean.npy')
x_test = x_test.reshape((x_test.shape[0],-1,x_test.shape[-1]))
x_test = np.delete(x_test,(57,58,59),axis = 1)
x_test = x_test.reshape((-1,38))
y_test = np.copy(x_test[:,15])
x = np.linspace(-89.5,90,720)
y = np.linspace(-179.5,180,1440)
y_test_mean = np.mean(y_test)
name = ['pftCrop','GDP','Distance_to_populated_area','WDPA_fracCover','road_density','Livestock','pop_density']
echoes = x_test[:,[6,14,16,27,34,35,37]].copy()
space_time = np.copy(x_test[:,[0,1,2]])
lon = np.reshape(space_time[:,1],(-1,57))[:,0]
lat = np.reshape(space_time[:,0],(-1,57))[:,0]
err = np.zeros((7,1))
err_mean = np.zeros((7,1))
x_test = np.delete(x_test,[0,1,2,6,14,15,16,18,19,21,22,23,24,25,26,27,32,34,35,36,37],axis = 1)
#x_train = np.delete(x_train,[0,1,2,6,14,15,18,19,21,22,23,24,25,26,32,34,35,36,37],axis = 1)
for i in range(echoes.shape[1]):
    plot_feature(i)
# np.save('../output/month_mean/err.npy',err)
# np.save('../output/month_mean/err_mean.npy',err_mean)