import numpy as np
from sklearn.ensemble import RandomForestRegressor
from multiprocessing import Pool
import joblib

def rf_test(eco):
    xtrain = np.append(x_train,np.transpose([echoes[:,eco]]),axis = 1)
    model = RandomForestRegressor (n_estimators=500, criterion='squared_error', random_state=42, max_features='', n_jobs=60,
                                   oob_score=True)
    model.fit(xtrain,y_train)
    joblib.dump(model, "../output/feature%s.joblib"%str(eco))

if __name__ == '__main__':
    global x_train
    x_train = np.load('../data/train_median.npy')
    global y_train
    y_train = x_train[:,15].copy()
    global echoes
    echoes = x_train[:,[34,35,37,14]].copy()
    #0: 34 Livestock
    #1: 35 road_density
    #2: 37 pop_density
    #3: 14 GDP
    x_train = np.delete(x_train,[2,14,15,18,19,21,22,23,24,25,26,34,35,36,37],axis = 1)
    with Pool() as p:
        p.map(rf_test,np.array([0,1,2,3]))
    p.close()
    p.join()
    #rf_test(None)
    # #Save model to predict and visualize"""  """