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
            self.y_n_categories = self.unique_y.shape[0]
        self.samples, self.n_predictors = self.x.shape

    def run(self):
        def _sd(j: int, vc: tuple):
            mat, i, unqy, xx = vc
            c = jnp.where(xx == unqy)[0]
            mat = mat.at[j, i].set(c)
            return mat, i, unqy, xx

        def _chisquared_lone(i: int, val: tuple):
            matrix, itr, unique_x, x_n = val
            unq_y = self.unique_y[i]
            jax.lax.fori_loop(lower=0, upper=x_n, body_fun=_sd, init_val=(matrix, i, unq_y, self.x[:, itr]))
            for j in range(x_n):
                mat, i, unqy, xx = _sd(j=j, vc=(matrix,i,unq_y,self.x[:, itr]))

            return

        def _chisquared_all(itr: int, values: tuple):
            unique_x = jnp.unique(self.x[:, itr])
            x_n_categories = unique_x.shape[0]
            contigency_matrix = jnp.zeros((x_n_categories, self.y_n_categories))  # rows -> x categ, column ->y categ
            for i in range(self.y_n_categories):
                _chisquared_lone(i=i, val=(contigency_matrix, itr, unique_x, x_n_categories))

            # jax.lax.fori_loop(lower=0,
            #                   upper=self.y_n_categories,
            #                   body_fun=_chisquared_lone,
            #                   init_val=(contigency_matrix, itr, unique_x, x_n_categories))

            # cTotal = obsCount.sum(axis=1)
            # rTotal = obsCount.sum(axis=0)

            return

        for itr in range(self.n_predictors):
            vl = _chisquared_all(itr, values=None)

        # jax.lax.fori_loop(lower=0,
        #                   upper=self.n_predictors,
        #                   body_fun=_chisquared_all,
        #                   init_val=None)




data = pd.read_csv('winequality-white.csv', delimiter=';')
x_0 = jnp.array(data.iloc[:, :-4].values)
x_0 = jnp.round(x_0[:, :6])
y_0 = jnp.round(x_0[:, 7])

DD = Fscchi2(x=x_0,y=y_0).run()