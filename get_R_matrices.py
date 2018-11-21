import os
import pprint
import numpy as np
import pickle

import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.packages import STAP
rpy2.robjects.numpy2ri.activate()
Rsession = rpy2.robjects.r

TSclust = importr('TSclust')

mfunc = 'myasmatrix <- function(dobj){return(as.matrix(dobj))}'
myasmatrix = STAP(mfunc, "myasmatrix")

mfunc = 'saxconv <- function(x,asize){return(convert.to.SAX.symbol( x, alpha=asize ))}'
saxconv = STAP(mfunc, "saxconv")

def znorm(x):
    return (x - x.mean()) / x.std()

def compute_and_save_metrics_R(data,n,dname,dirname,njobs=40):
    # Creare a SAX symbolic representation with alphabet size 10
    Xsax = [np.asarray(saxconv.saxconv(znorm(_x),10)) for _x in data]
    Xsax = np.vstack(Xsax)
    

    # p
    # If not NULL, sets the weight for the geometric decaying of the autocorrelation coefficients. 
    # Ranging from 0 to 1.
    for _p in [0.05,0.1]:
        rres = Rsession.diss( data , "ACF", p=_p)
        X = np.array(myasmatrix.myasmatrix(rres))

        settings = {'dataset':dname,'method':'ACF','params':{'p':_p}}
        fname = '%s_%s_%f.pkl'%(dname,'ACF',_p)
        pickle.dump((X,settings),open(os.path.join(dirname,fname),'wb'),-1)


    # CDM distance
    rres = Rsession.diss(Xsax,"CDM",type="bzip2")
    X = np.array(myasmatrix.myasmatrix(rres))
    settings = {'dataset':dname,'method':'CDM','params':{'type':'bzip2'}}
    fname = '%s_%s_%s.pkl'%(dname,'CDM','bzip2')
    pickle.dump((X,settings),open(os.path.join(dirname,fname),'wb'),-1)
    
    # CID distance
    rres = Rsession.diss(data,"CID")
    X = np.array(myasmatrix.myasmatrix(rres))
    settings = {'dataset':dname,'method':'CID','params':None}
    fname = '%s_%s.pkl'%(dname,'CID')
    pickle.dump((X,settings),open(os.path.join(dirname,fname),'wb'),-1)
    
    # Correlation based distance
    rres = Rsession.diss(data,"COR")
    X = np.array(myasmatrix.myasmatrix(rres))
    settings = {'dataset':dname,'method':'COR','params':None}
    fname = '%s_%s.pkl'%(dname,'COR')
    pickle.dump((X,settings),open(os.path.join(dirname,fname),'wb'),-1)
    
    # Time correlation
    rres = Rsession.diss(data,"CORT")
    X = np.array(myasmatrix.myasmatrix(rres))
    settings = {'dataset':dname,'method':'CORT','params':None}
    fname = '%s_%s.pkl'%(dname,'CORT')
    pickle.dump((X,settings),open(os.path.join(dirname,fname),'wb'),-1)
    
    # Wavelet
    rres = Rsession.diss(data,"DWT")
    X = np.array(myasmatrix.myasmatrix(rres))
    settings = {'dataset':dname,'method':'DWT','params':None}
    fname = '%s_%s.pkl'%(dname,'DWT')
    pickle.dump((X,settings),open(os.path.join(dirname,fname),'wb'),-1)
    
    # FRECHET
    rres = Rsession.diss(data,"FRECHET")
    X = np.array(myasmatrix.myasmatrix(rres))
    settings = {'dataset':dname,'method':'FRECHET','params':None}
    fname = '%s_%s.pkl'%(dname,'FRECHET')
    pickle.dump((X,settings),open(os.path.join(dirname,fname),'wb'),-1)
    
    #NCD normalized compression
       
    rres = Rsession.diss(Xsax,"NCD",type="bzip2")
    X = np.array(myasmatrix.myasmatrix(rres))
    settings = {'dataset':dname,'method':'NCD','params':{'type':'bzip2'}}
    fname = '%s_%s_%s.pkl'%(dname,'NCD','bzip2')
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
    compute_and_save_metrics_R(pm25.values,pm25.shape[0],'pm25_2016','Data',njobs=40)
