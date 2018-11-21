import numpy as np
from multiprocessing import Pool
import pickle
import os
from sklearn.metrics import pairwise_distances
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.functions import SignatureTranslatedFunction
rpy2.robjects.numpy2ri.activate()
Rsession = rpy2.robjects.r
DTW = importr('dtw')

Rsession.dtw = SignatureTranslatedFunction(Rsession.dtw,init_prm_translate={'window_size': 'window.size'})

def interpolate(x,maxlen):
    if len(x) == maxlen:
        return x
    else:
        return np.interp(np.linspace(0, 1, maxlen),np.linspace(0, 1, len(x)), x)
    
def compute_dtw_R_sakoechiba(tup):
    s1,s2,wsize = tup    
    alignment = Rsession.dtw(gdata[s1],gdata[s2],keep_internals=False,distance_only=True,window_type='sakoechiba',window_size=wsize)
    val =alignment.rx('distance')[0][0]
    #val = alignment.rx('normalizedDistance')[0][0]
    return val

def compute_dtw_R_full(tup):
    s1,s2 = tup    
    alignment = Rsession.dtw(gdata[s1],gdata[s2],keep_internals=False,distance_only=True)
    val =alignment.rx('distance')[0][0]
    #val = alignment.rx('normalizedDistance')[0][0]
    return val

def compute_eucl(tup):
    s1,s2 = tup
    dist = np.linalg.norm(gdata[s1]- gdata[s2])        
    return dist

def compute_and_save_metrics_DTW(data,n,dname,dirname,njobs=80,winsizes=[7,24]):
    '''
    Function to compute pairwise DTW results with multiple cores.
    This is a separate function because it might take a long time to compute and we may
    run it on a separate server.
    '''
    global gdata
    gdata = data
    #upper triangular indices
    idxs = np.triu_indices(n,1)
    idxs_coords = zip(*idxs)
    pool = Pool(processes=njobs)
        
    #compute DTW
    for wlen in winsizes:
        iterable = [ (x[0],x[1],wlen) for x in idxs_coords]
        result = pool.map(compute_dtw_R_sakoechiba, iterable)
        X= np.zeros((n,n))                                                                                                                                    
        X[idxs]=np.array(result)                                            
        X[np.tril_indices(n,-1)]=X.T[np.tril_indices(n,-1)] 
        settings = {'dataset':dname,'method':'DTW','params':{'sakoechiba_window_size':wlen}}
        fname = '%s_%s_%d.pkl'%(dname,'DTW',wlen)
        pickle.dump((X,settings),open(os.path.join(dirname,fname),'wb'),-1)

    #compute full DTW    
    iterable = [ (x[0],x[1]) for x in idxs_coords]
    result = pool.map(compute_dtw_R_full, iterable)
    X= np.zeros((n,n))                                                                                                                                    
    X[idxs]=np.array(result)                                            
    X[np.tril_indices(n,-1)]=X.T[np.tril_indices(n,-1)] 
    settings = {'dataset':dname,'method':'DTW','params':None}
    fname = '%s_%s_full.pkl'%(dname,'DTW')
    pickle.dump((X,settings),open(os.path.join(dirname,fname),'wb'),-1)
    pool.close()    
    
    
def compute_and_save_metrics_others(data,n,dname,dirname,njobs=40):
    #Euclidean
    X = pairwise_distances(data,  metric='euclidean', n_jobs=njobs)
    settings = {'dataset':dname,'method':'euclidean','params':None}
    fname = '%s_%s.pkl'%(dname,'euclidean')
    pickle.dump((X,settings),open(os.path.join(dirname,fname),'wb'),-1)


if __name__=='__main__':                                                                                                                                                                                    
    import pandas as pd                                                                                                                                                                                     
    pm25 = pd.read_csv('Data/pm25_2016.csv')                                                                                                                                                                
    rowindex = pm25[pm25.columns[[0]]]                                                                                                                                                                      
    # drop station column                                                                                                                                                                                   
    pm25.drop(pm25.columns[[0]], axis=1, inplace=True)                                                                                                                                                      
    # first, fill forward                                                                                                                                                                                   
    pm25.fillna(method='ffill',axis=1,inplace=True)                                                                                                                                                         
    # then fill backward                                                                                                                                                                                    
    pm25.fillna(method='bfill',axis=1,inplace=True)
    compute_and_save_metrics_others(pm25.values,pm25.shape[0],'pm25_2016','Data',njobs=64)
    compute_and_save_metrics_DTW(pm25.values,pm25.shape[0],'pm25_2016','Data',njobs=64)
