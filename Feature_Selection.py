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

        self.chi_squared_statistics = jnp.zeros((self.n_predictors,))
        self.chi_squared_p_values = jnp.zeros((self.n_predictors,))
        self.cramer_v = jnp.zeros((self.n_predictors,))

    def run(self):

        def inner_loop(m, values3):
            unqqx, feat_cnt, i, mat = values3
            ff = jnp.where((self.x[:, feat_cnt] == unqqx[m]) & (self.y == self.unique_y[i]))[0]
            mat = mat.at[m, i].set(ff.shape[0])
            return unqqx, feat_cnt, i, mat

        def over_y_categories(i, values2):
            mat, feat_cnt, unqx, x_n = values2
            for m in range(x_n):
                unqx, feat_cnt, i, mat = inner_loop(m=m, values3=(unqx, feat_cnt, i, mat))
            return mat, feat_cnt, unqx, x_n

        def over_features(feat_cnt, values_over_features):
            chi_squared_statistics, chi_squared_p_values, cramer_v = values_over_features


            unique_x = jnp.unique(self.x[:, feat_cnt])
            x_n_categories = unique_x.shape[0]
            contigency_matrix = jnp.zeros((x_n_categories, self.y_n_categories))  # rows -> x categ, column ->y categ
            for i in range(self.y_n_categories):
                mat, feat_cnt, unqx, x_n = over_y_categories(i=i, values2=(
                contigency_matrix, feat_cnt, unique_x, x_n_categories))
            return chi_squared_statistics, chi_squared_p_values, cramer_v

        self.chi_squared_statistics = jnp.zeros((self.n_predictors,))
        self.chi_squared_p_values = jnp.zeros((self.n_predictors,))
        self.cramer_v = jnp.zeros((self.n_predictors,))


        for feat_cnt in range(self.n_predictors):
            chi_squared_statistics, chi_squared_p_values, cramer_v = over_features(feat_cnt=feat_cnt,
                                                                                   values_over_features=(
                                                                                    self.chi_squared_statistics.copy(),
                                                                                    self.chi_squared_p_values.copy(),
                                                                                    self.cramer_v.copy()))


data = pd.read_csv('winequality-white.csv', delimiter=';')
x_0 = jnp.array(data.iloc[:, :-4].values)
x_0 = jnp.round(x_0[:, :6])
y_0 = jnp.round(x_0[:, 7])

DD = Fscchi2(x=x_0, y=y_0).run()
