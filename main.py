import os
import numpy as np
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen

DATA_PATH = 'Data/'

def download_data():
    data_url = 'http://timeseriesclassification.com/Downloads/ItalyPowerDemand.zip'
    resp = urlopen(data_url)
    zipfile = ZipFile(BytesIO(resp.read()))
    #fnames = zipfile.namelist()
    data = np.array([list(map(float, line.strip().split())) 
                     for line in zipfile.open('ItalyPowerDemand_TRAIN.txt').readlines()])
    xtrain, ytrain = data[:,1:],data[:,0]
    data = np.array([list(map(float, line.strip().split())) 
                     for line in zipfile.open('ItalyPowerDemand_TEST.txt').readlines()])    
    xtest, ytest = data[:,1:],data[:,0]
    return xtrain, ytrain, xtest, ytest

if __name__ == "__main__":
    fname = "ItalyPowerDemand.npy"
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    if not os.path.isfile(os.path.join(DATA_PATH,fname)):
        xtrain, ytrain, xtest, ytest = download_data()
        np.save(open(os.path.join(DATA_PATH,fname),'wb'),[xtrain, ytrain, xtest, ytest])
    else:
        xtrain, ytrain, xtest, ytest = np.load(open(os.path.join(DATA_PATH,fname),'rb'))