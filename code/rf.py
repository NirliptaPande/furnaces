import numpy as np
from sklearn.ensemble import RandomForestRegressor
#from multiprocessing import Pool
import joblib
#def rf_test(eco):
# def rf_test(eco):
    # if (eco is None):
    #     pass
    # else:
    #     xtrain = np.append(x_train,eco)
   
if __name__ == '__main__':
    #global x_train
    x_train = np.load('../data/train_median.npy')
    #global y_train
    y_train = x_train[:,15].copy()
    # eco = x_train[:,[34,35,37,14]].copy()
    x_train = np.delete(x_train,[2,14,15,18,19,21,22,23,24,25,26,34,35,36,37],axis = 1)
    model = RandomForestRegressor (n_estimators=500, criterion='squared_error', random_state=42, max_features=7, n_jobs=60,
                                   oob_score=True)
    model.fit(x_train,y_train)
    joblib.dump(model, "../output/none.joblib")
    #rf_test(None)
    # #Save model to predict and visualize"""  """