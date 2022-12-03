import pandas as pd
import numpy as np
import sklearn
from sklearn import decomposition


data = pd.read_csv('winequality-white.csv', delimiter=';')
data = np.array(data.values[:, :-2])

# T = FactorAnalysis_(x=data, n_comp=2, tolerance=1e-6, max_iter=1000, random_seed=1)
# T.calculate()

# data = ((data - data.mean(axis=0))/data.std(axis=0))

pp = decomposition.FactorAnalysis(n_components=2,max_iter=10000,tol=1e-3,svd_method='lapack')
DD = pp.fit(data)
DD