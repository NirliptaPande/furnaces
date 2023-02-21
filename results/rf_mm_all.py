import numpy as np
import joblib
# from mpl_toolkits.basemap import Basemap
# import matplotlib.pyplot as plt
import gc
x_test  = np.load('../data/test_monthmean.npy')
x_test = x_test.reshape((x_test.shape[0],-1,x_test.shape[-1]))
x_test = np.delete(x_test,(57,58,59),axis = 1)
x_test = x_test.reshape((-1,38))
y_test = np.copy(x_test[:,15])
space_time = np.copy(x_test[:,[0,1,2]])
lon = np.reshape(space_time[:,1],(-1,57))[:,0]
lat = np.reshape(space_time[:,0],(-1,57))[:,0]
y_test_mean = np.mean(y_test)
err_all_none = np.array([0.0,0.0,0.0,0.0])#all_nmse,all_mean, none_nmse,none_mean
x_test = np.delete(x_test,[0,1,2,15,18,19,21,22,23,24,25,26,32,36],axis = 1)
model = joblib.load('../output/month_mean/feature_all.joblib')
y_pred = model.predict(x_test)
np.save('../output/month_mean/y_pred_all.npy',y_pred)
# err_all_none[0] = np.divide(np.mean(((y_test-y_pred)**2)),np.mean(y_test**2))
# y_pred_mean = np.mean(np.reshape(y_pred,(-1,57)),axis = 1)
# err_all_none[1] = np.divide(np.mean(((y_test_mean-np.mean(y_pred))**2)),y_test_mean**2)
# # x = np.linspace(-89.5,90,720)
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
# plt.title(" Predicted Ignitions - all")
# plt.colorbar()
# plt.savefig('../output/month_mean/all')
# plt.close()
del(model)
gc.collect()

x_test = np.delete(x_test,[3,11,12,15,21,22,23],axis = 1)
model = joblib.load('../output/month_mean/feature_none.joblib')
y_pred = model.predict(x_test)
np.save('../output/month_mean/y_pred_none.npy',y_pred)
# err_all_none[2] = np.divide(np.mean(((y_test-y_pred)**2)),np.mean(y_test**2))
# y_pred_mean = np.mean(np.reshape(y_pred,(-1,57)),axis = 1)
# err_all_none[3] = np.divide(np.mean(((y_test_mean-np.mean(y_pred))**2)),y_test_mean**2)
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
# plt.title(" Predicted Ignitions - none")
# plt.colorbar()
# plt.savefig('../output/month_mean/none.png')
# plt.close()
# np.save('../output/month_mean/err_all_none.npy',err_all_none)