import jax.numpy as jnp
import jax
import pandas as pd


class Fscchi2:
    def __init__(self,
                 x: jnp.ndarray = None,
                 y: jnp.ndarray = None):
        'Univariate feature ranking for classification using chi-square tests'

        if isinstance(x, jnp.ndarray):
            if jnp.any(jnp.isnan(x)):
                raise Exception('The matrix of predictors contain NaN values!')
            else:
                self.x = x


        if isinstance(y, jnp.ndarray):
            if jnp.any(jnp.isnan(y)):
                raise Exception('The target variable contains NaN values!')
            else:
                self.y = y

        self.unique_y = jnp.unique(self.y)
        if self.unique_y.shape[0] < 2:
            raise Exception('The target variable is made of only one categorical variable.')
        else:
            self.y_n_ = self.unique_y.shape[0]
        self.samples, self.n_predictors = self.x.shape
        self.chi_squared_statistics = jnp.zeros((self.n_predictors,))
        self.chi_squared_p_values = jnp.zeros((self.n_predictors,))
        self.cramer_v = jnp.zeros((self.n_predictors,))
        self.contigency_matrix = jnp.zeros((self.samples, self.y_n_, self.n_predictors))  # (samples, n_y_categories, p)
    def run(self):

        def inner_loop(x_cat_cnt, values_inner_loop):
            unique_x, feat_cnt, y_cat_cnt, contigency_matrix = values_inner_loop
            counts = jnp.where((self.x[:, feat_cnt] == unique_x[x_cat_cnt]) & (self.y == self.unique_y[y_cat_cnt]))[0]
            contigency_matrix = contigency_matrix.at[x_cat_cnt, y_cat_cnt].set(counts.shape[0])
            return unique_x, feat_cnt, y_cat_cnt, contigency_matrix

        def over_y_categories(y_cat_cnt, values_over_y_catg):
            contigency_matrix, feat_cnt, unique_x, x_n_ = values_over_y_catg
            for x_cat_cnt in range(x_n_):
                unique_x, feat_cnt, y_cat_cnt, contigency_matrix = inner_loop(x_cat_cnt=x_cat_cnt,
                                                                              values_inner_loop=(unique_x,
                                                                                                 feat_cnt,
                                                                                                 y_cat_cnt,
                                                                                                 contigency_matrix))
            return contigency_matrix, feat_cnt, unique_x, x_n_

        def over_features(feat_cnt, values_over_features):
            contigency_matrix = values_over_features

            unique_x = jnp.unique(self.x[:, feat_cnt])
            x_n_ = unique_x.shape[0]

            for y_cat_cnt in range(self.y_n_):
                contigency_matrix, feat_cnt, unique_x, x_n_ = over_y_categories(y_cat_cnt=y_cat_cnt,
                                                                                values_over_y_catg=(contigency_matrix,
                                                                                                    feat_cnt,
                                                                                                    unique_x,
                                                                                                    x_n_))

            ####
            return contigency_matrix

        for feat_cnt in range(self.n_predictors):
            chi_squared_statistics, chi_squared_p_values, cramer_v = over_features(feat_cnt=feat_cnt,
                                                                                   values_over_features=self.contigency_matrix.copy())



data = pd.read_csv('winequality-white.csv', delimiter=';')
x_0 = jnp.array(data.iloc[:, :-4].values)
x_0 = jnp.round(x_0[:, :6])
y_0 = jnp.round(x_0[:, 7])


import numpy as np


obsCount = pd.crosstab(index=x_0[:,1], columns=y_0, margins=False, dropna=True)
cTotal = obsCount.sum(axis=1)
rTotal = obsCount.sum(axis=0)
nTotal = np.sum(rTotal)
expCount = np.outer(cTotal, (rTotal / nTotal))
chiSqStat = ((obsCount - expCount) ** 2 / expCount).to_numpy().sum()
chiSqDf = (obsCount.shape[0] - 1.0) * (obsCount.shape[1] - 1.0)
chiSqSig = scipy.stats.chi2.sf(chiSqStat, chiSqDf)



DD = Fscchi2(x=x_0, y=y_0).run()
