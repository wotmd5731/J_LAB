# -*- coding: utf-8 -*-

from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# We would like all available data from 01/01/2000 until 12/31/2016.
start_date = '2018-01-01'
end_date = '2019-01-01'

# User pandas_reader.data.DataReader to load the desired data. As simple as that.
#source = data.DataReader('005930.KS', 'yahoo', start_date, end_date)#삼성전자
#source = data.DataReader('207940.KS', 'yahoo', start_date, end_date)#바이오로직스
source = data.DataReader('006400.KS', 'yahoo', start_date, end_date)#sdi

#%%
adj_data = source['Adj Close']
data = source['Close']

#ret = data.pct_change(1)
log_ret = np.log(data/data.shift(1))

def min_max_norm(wdata):
    return ( wdata[-1] - wdata.min() )/ (wdata.max()-wdata.min())
def mean_std_norm(wdata):
    return ( wdata[-1] - wdata.mean() )/ wdata.std()


def svd_whiten(X):
#    a = source[X]
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    # U and Vt are the singular matrices, and s contains the singular values.
    # Since the rows of both U and Vt are orthonormal vectors, then U * Vt
    # will be white
    X_white = np.dot(U, Vt)
    return X_white[-1]

def rolling_whiten(src,window=20,min_periods=10):
    ret = []
    for i in range(src.__len__()):
        if i < min_periods-1:
            ret.append([0 for i in range(src.columns.__len__())])
        elif i<window:
            ret.append(svd_whiten(src[0:i].values))
        else:
            ret.append(svd_whiten(src[i-window:i].values))
    pdata = pd.DataFrame(np.stack(ret,0))
    pdata.columns = ['w'+str(i) for i in range(src.columns.__len__())]
    pdata.index = src.index
    return pdata



norm0 = data.rolling(window=30,min_periods=10).apply(min_max_norm,raw=True).rename('minmax')
norm1 = data.rolling(window=30,min_periods=10).apply(mean_std_norm,raw=True).rename('stdard')
norm2 = rolling_whiten(source,window=30,min_periods=10)#pca whitening using close open min max vol

    
aa= pd.concat([norm0,norm1,norm2],axis=1)



#ma60 = data.rolling(window=60).mean()
#ma30 = data.rolling(window=30).mean()

#data.plot()
#ma30.plot()
#ma60.plot()
#action = ma30 - ma60
#action[action < 0 ] =-1
#action[action > 0 ] =100000
#
#action.plot()
#
#print('total benefit :',log_ret[action>0].sum())


#log_ret.hist(bins=50)

#data.rolling(30).mean().plot()
#data.rolling(30).var().plot()







