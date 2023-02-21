from multiprocessing import Pool
import numpy as np
import gc
global arr
def split_test(index):
    step = int(arr.shape[0]/720/1440)
    np.save('../../data/new/test_%s.npy'%index,arr[index*14400*step:((index+1)*14400*step)])
def split_train(index):
    step = int(arr.shape[0]/720/1440)
    np.save("../../data/new/train_%s.npy"%index,arr[index*14400*step:((index+1)*14400*step)])

if __name__ == '__main__':
    indices = np.arange(72)
    arr = np.load('../../data/test/test.pkl', allow_pickle='True')
    with Pool() as p:
        p.map(split_test,indices)
    p.close()
    p.join()
    del(arr)
    gc.collect()
    arr = np.load('../../data/test/train.pkl', allow_pickle='True')
    with Pool() as p:
        p.map(split_train,indices)
    p.close()
    p.join()
    del(arr)
    gc.collect()