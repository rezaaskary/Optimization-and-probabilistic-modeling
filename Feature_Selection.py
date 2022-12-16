import jax.numpy as jnp
import jax


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

        def _chisquared_lone(i: int, val: tuple):


            return

        def _chisquared_all(itr: int, values: tuple):
            unique_x = jnp.unique(self.x[:, itr])
            x_n_categories = unique_x.shape[0]
            contigency_matrix = jnp.zeros((x_n_categories, self.y_n_categories))  # rows -> x categ, column ->y categ
            jax.lax.fori_loop(lower=0,
                              upper=self.y_n_categories,
                              body_fun=_chisquared_lone,
                              init_val=(contigency_matrix, ))

            cTotal = obsCount.sum(axis=1)
            rTotal = obsCount.sum(axis=0)



            return

        jax.lax.fori_loop(lower=0,
                          upper=self.n_predictors,
                          body_fun=_chisquared_all,
                          init_val=None)

        jax.scipy.stats.contingency.crosstab
