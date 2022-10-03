from multiprocessing import Pool
import numpy as np
import gc
global arr
def split_test(index):
    step = int(arr.shape[0]/720/1440)
    np.save('../data/test_%s.npy'%index,arr[index*14400*step:((index+1)*14400*step)-1])
def split_train(index):
    step = int(arr.shape[0]/720/1440)
    np.save('../data/train_%s.npy'%index,arr[index*14400*step:((index+1)*14400*step)-1])

if __name__ == '__main__':
    indices = np.arange(72)
    #global arr
 #   arr = np.load('../data/test.pkl', allow_pickle='True')
 #   with Pool() as p:
 #       p.map(split_test,indices)
 #   p.close()
 #   p.join()
 #   del(arr)
 #   gc.collect()
    arr = np.load('../data/test.pkl', allow_pickle='True')
    with Pool() as p:
        p.map(split_test,indices)
    p.close()
    p.join()
    del(arr)
    gc.collect()
