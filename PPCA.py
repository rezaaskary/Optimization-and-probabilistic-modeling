import numpy as np
import scipy as sc
import pandas as pd
import warnings
import sys
from IPython.display import display,clear_output
from sklearn.decomposition import PCA 
from sklearn.impute import SimpleImputer

def emppca_complete(Y,k,W,v,MaxIter,TolFun,TolX,dispnum,iterfmtstr):
    p,n = Y.shape
    mu = np.mean(Y,axis=1)[:,np.newaxis]
    Y -= np.tile(mu,[1,n])
    iter = 0
    nloglk = np.inf
    traceS = ((Y.reshape((-1,1))).T@Y.reshape((-1,1)))/(n-1)
    eps = np.finfo(float).eps
    while iter < MaxIter:
        iter +=1
        SW = Y@(Y.T@W)/(n-1)
        M = W.T@W + v*np.eye(k)
        Wnew = SW@np.linalg.inv(v*np.eye(k) + np.linalg.inv(M)@W.T@SW)
        vnew = (traceS - np.trace(SW@np.linalg.inv(M)@Wnew.T))/p   
        
        dw = (np.abs(W-Wnew) / (np.sqrt(eps)+(np.abs(Wnew)).max())).max()
        dv = np.abs(v-vnew) / (eps+v)
        delta = max([dw,dv])
        CC = Wnew@Wnew.T + vnew*np.eye(p)
        nloglk_new = (p*np.log(2*np.pi) + np.log(np.linalg.det(CC)) +\
             np.trace(np.linalg.inv(CC)@Y@Y.T/(n-1)) )*n/2
        W = Wnew
        v = vnew
        print(delta)
        print(np.abs(nloglk - nloglk_new))
        if delta < TolX:
            break
        elif (nloglk - nloglk_new) < TolFun:
            break
        elif np.abs(vnew) < np.sqrt(eps):
            break
        nloglk = nloglk_new
    ##=====================================================
    Xmu = np.linalg.inv(M)@Wnew.T@Y
    return Wnew,Xmu,mu,vnew,iter,dw,nloglk_new
#######################################################################################


def PPCA(Y: np.ndarray, k:int=2):
    """

    """
    if not (min(Y.shape)>1): raise Exception('Too few feature variables')
    Y = Y.T
    wasNaN = pd.isnull(Y)
    hasMissingValue = np.any(wasNaN)
    allNaN = np.all(wasNaN,axis=0)
    Y = Y[:,~allNaN]
    wasNaN = wasNaN[:,~allNaN]
    obs = ~wasNaN
    numObs = obs.sum()
    p,n = Y.shape
    ####################################################
    if ~ Y.any():
        coeff = np.nan
        score = np.nan
        latent = np.nan
        mu = np.nan
        v = np.nan
        rsltStruct = np.nan
        return coeff, score, latent, mu, v, rsltStruct
    
    ######################################################
    maxRank = min([n,p])
    flagWarnK = False

    if (not np.isscalar(k)) or ( not isinstance(k, int)):
        raise Exception('invalid k!')
    elif k > maxRank-1:
        k = max([1, maxRank -1])
        flagWarnK = True
        print(f'Warning: Maximum possible rank of the data is {maxRank}. Computation continues with the number of principal components k set to {k}')
    else:
        pass
    #########################################################
    paramNames = ['Options', 'W0',  'v0']
    setFlag = dict(zip(paramNames,[0, 0, 0]))
    Opt = [None]
    W = 0.01*np.random.normal(0, 1, p*k).reshape((p,k))
    v = np.random.rand()
    ######################################################
    TolX = 1e-6
    TolFun = 1e-6
    MaxIter = 2e4
    DispOpt = 'off'
    if (setFlag['W0']) and flagWarnK and W.shape[1] == maxRank:
        W = W[:,:-1] # remove the last column
        print(f'Warning: Initial value W0 is truncated to {n} by {k}')
    
    if (setFlag['W0']) and (np.any(pd.isnull(W))):
        raise Exception(f'Initial matrix W0 must be a {p} by {k} numeric matrix without any NaN element')
    
    if (setFlag['v0']) and (not(np.isscalar(k) and v>0) or pd.isnull(v) or n == np.inf):
        raise Exception('Initial residual variance v0 must be a positive scalar and must not be Inf.')
    
    ##################Supress Warnings##########################
    if not sys.warnoptions:
        warnings.simplefilter('ignore')
    #########################################################
    # Preallocate memory
    mu = np.zeros((p,1))
    X = np.zeros((k,n))
    Wnew = np.zeros((p,k))
    C = np.zeros((k,k,n))
    nloglk = np.inf

    dispnum = [1,0,0]
    headernames = ['Iteration      Variance     |Delta X|      Negative Log-likelihood']
    if dispnum[1]:
        print(headernames)

    itercount = 0
    if hasMissingValue:
        # If Y has any missing value, use the following algorithm
        while itercount < MaxIter:
            itercount+=1
            for j in range(n):
                y = Y[:,j:j+1]
                idxObs = obs[:,j]
                w = W[idxObs,:]
                # Use Sherman-Morrison formula to find the inv(v.*eye(k)+w'*w)
                Cj = np.eye(k)/v-(w.T@w)*np.linalg.inv(np.eye(k)+(w.T@w)/v)/(v**2)
                # Cj = np.linalg.inv(v*np.eye(k) + w.T@w)
                X[:,j:j+1] = Cj@(w.T@(y[idxObs]-mu[idxObs]))
                C[:,:,j] = Cj
                ##=========================================================================
            mu = np.nanmean(Y-W@X,axis=1)[:,np.newaxis]

            for i in range(p):
                idxObs = obs[i,:]
                M = X[:,idxObs]@X[:,idxObs].T + v*np.sum(C[:,:,idxObs],axis=2)
                ww = X[:,idxObs]@(Y[i,idxObs]-mu[i,0]).T
                Wnew[i,:] = np.linalg.solve(M,ww)
            
            vsum = 0
            for j in range(n):
                wnew = Wnew[obs[:,j],:]
                vsum = vsum + sum((Y[obs[:,j],j] - wnew@X[:,j] - mu[obs[:,j],0])**2+\
                    v * (np.diag(wnew@C[:,:,j]@wnew.T)))
            vnew = vsum/numObs
            eps = np.finfo(float).eps
            nloglk_new = 0
            for j in range(n):
                idxObs = obs[:,j]
                y = Y[idxObs,j:j+1]  - mu[obs[:,j],0:1]
                Wobs = Wnew[idxObs,:]
                Cy = Wobs@Wobs.T + vnew*np.eye(sum(idxObs))
                nloglk_new = nloglk_new + (sum(idxObs)*np.log(2*np.pi)  + np.log(np.linalg.det(Cy)) + np.trace(np.linalg.inv(Cy)@y@y.T))/2
                
            dw =(np.abs(W-Wnew) / (np.sqrt(eps) + (np.abs(Wnew)).max())).max()

            W = Wnew
            v = vnew

            # if dw<TolX:
            #     break
            # elif (nloglk-nloglk_new) < TolFun:
            #     break
            # nloglk = nloglk_new
            print(np.abs(nloglk-nloglk_new))
            if np.abs(nloglk-nloglk_new) < TolFun:
                break
            nloglk = nloglk_new


        muX = np.mean(X,axis=1)[:,np.newaxis]
        X = X - np.tile(muX,[1,n])
        mu = mu + W@muX

    else:
        iterfmtstr = ''
        W,X,mu,v,itercount,dw,nloglk  = emppca_complete(Y,k,W,v,MaxIter,TolFun,TolX,dispnum,iterfmtstr)
        
        if np.all(W==0):
            coeff = np.zeros((p,k))
            coeff[::(p+1)] = 1
            score = np.zeros((n,k))
            latent = np.zeros(k,1)
            mu = (np.mean(Y,axis=1)).T
            v = 0

            
            rsltStruct = {'W':W,\
                        'Xexp':X.T,\
                        'Recon':np.tile(mu,[n,1]),\
                        'v':v,\
                        'NumIter':itercount,\
                        'RMSResid':0,\
                        'nloglk':nloglk
            }
            return coeff, score, latent, mu, v, rsltStruct
################################################################################
    # Reconstruction:
    WTW = W.T@W
    Y_hat = (W@np.linalg.inv(WTW)) @ (WTW + v*np.eye(k))@X
    diff = Y-Y_hat - np.tile(mu,[1,n])
    diff[~obs]=0
    rmsResid = np.linalg.norm(diff)/np.sqrt(numObs)
    print(f'Iteration: {itercount}   Variance: {v}    |Delta X|: {dw}   Negative Log-likelihood: {nloglk}')
    coeff, _, _ = np.linalg.svd(W,full_matrices=False)
    score = Y_hat.T@coeff
    latent = np.real(np.linalg.eigvals(score.T@score))
    latent = ((np.sort(latent)/(n-1))[::-1])[:,np.newaxis]
    mu = mu.T
    ##==============================================================================
    # Enforce a sign convention on the coefficients -- the largest element in
    # each column will have a positive sign.
    maxind = np.argmax(np.abs(coeff),axis=0)

    (d1,d2) = coeff.shape
    colsign = np.sign(coeff[maxind,np.arange(0,k)])
    coeff = coeff * np.tile(colsign,[p,1])

    score = score * np.tile(colsign,[n,1])
    score = np.insert(score,np.where(allNaN)[0] ,values = np.nan,axis=0)
    

    rsltStruct = {  'W':W,
                    'Xexp':X.T,
                    'Recon':Y_hat.T + np.tile(mu,[n,1]),\
                    'v':v,\
                    'NumIter': itercount,\
                    'RMSResid': rmsResid,\
                    'nloglk': nloglk}


    return coeff, score, latent, mu, v, rsltStruct




dat = pd.read_csv('DFD.csv',header=None)
dat = (dat.values).astype('float')
Y=dat

k=4
# dat[14,2]= np.nan
# dat[45,1]= np.nan


coeff, score, latent, mu, v, rsltStruct= PPCA(dat, k=k)
