import numpy as np
class Liklihood_Functions:
    def __int__(self,function:str = 'gaussian_single_variable', vectorized:bool = False,\
                sigma:np.ndarray = np.ones((1,1)), Covariance: np.ndarray = np.ones((1,1)),\
                K:int = None, measured: np.ndarray = None, estimated: np.ndarray = None,
                C: int = None, diagonal_covariance:bool = None):

        """
        :param function: A string variable indicating the type of liklihood function
        :param vectorized:
        :param Covariance: A positive definite square matrix indicating the covariance matrix of the multivariable Normal distribution (KxK)
        :param K: The dimention of the multivariable gaussian distribution
        :param measured: The measured variable
        :param estimated: The estimated variable(or calculated from a model)
        :param C: An integer indicating the number of Chains
        :param diagonal_covariance:
        :param sigma: The standard deviation of the error estimation
        :return: the liklihood value(s)
        """



        if not isinstance(vectorized, bool):
            self.vectorized = False
            print(
                f'------------------------------------------------------------------------------------------------------------------\n '
                f'The default value of {self.vectorized} is selectd for vectorizing simulations\n'
                f'---------------------------------------------------------------------------------------------------------------------')
        else:
            self.vectorized = vectorized


        if not isinstance(C, int):
            self.C = 1
            print(
                f'------------------------------------------------------------------------------------------------------------------\n '
                f'The default value of {self.C} is selectd for the number of chains\n'
                f'---------------------------------------------------------------------------------------------------------------------')
        else:
            self.C = C


        if not isinstance(K, int):
            raise Exception('The dimention of the problem is not specified correctly!')
        else:
            self.K = K


        if not isinstance(N, int):
            raise Exception('The size of the data is not specified correctly!')
        else:
            self.N = N


        if not isinstance(diagonal_covariance, bool):
            raise Exception('The type of calculation(vectorizing) is not specified correctly!')
        else:
            self.Diagonal = diagonal_covariance


        if not isinstance(Covariance, np.ndarray):
            raise Exception('The covariance matrix is not specified correctly!')
        else:
            self.Covariance = Covariance


        if not isinstance(sigma, (int, float)):
            raise Exception('The standard deviation is not specified correctly!')
        else:
            self.sigma = sigma

            if not isinstance(sigma, (int, float)):
                raise Exception('The standard deviation is not specified correctly!')
            else:
                self.sigma = sigma

        if not isinstance(measured, (int, np.ndarray)):
            raise Exception('the measured data is not entered correctly!')
        else:
            self.measured = measured

        if not isinstance(estimated, (int, np.ndarray)):
            raise Exception('the measured data is not entered correctly!')
        else:
            self.estimated = estimated

        if isinstance(function, str):
            raise Exception('the type of liklihood function is not specified correctly!')
        else:
            self.function = function

        if self.function is 'gaussian_single_variable' and not self.vectorized:
            self.liklihood = gaussian_liklihood_single_variable

        elif self.function is 'gaussian_single_variable' and  self.vectorized:

            self.liklihood = gaussian_liklihood_single_variable_vectorized

        elif self.function is 'multivariable_gaussian' and not self.vectorized:
            self.liklihood = gaussian_liklihood_multivariable

        elif self.function is 'multivariable_gaussian' and self.vectorized:
            self.liklihood = gaussian_liklihood_multivariable_vectorized

        else:
            raise Exception('The type of entered lklihood function is not implemented!')


    def gaussian_liklihood_single_variable(self) -> np.ndarray:
        """
        The single variable Gausian liklihood function
        :param measured: The measured variable
        :param estimated: The estimated variable(or calculated from a model)
        :param N: The number of measured samples
        :param sigma: The standard deviation of the error estimation
        :return: the log_liklihood function
        """

        log_gauss = -self.N*np.log(self.sigma * np.sqrt(2 * np.pi)) - (((self.measured - self.estimated) ** 2) / (2 * self.sigma ** 2)).sum()
        return log_gauss
        #====================================================================================

    def gaussian_liklihood_single_variable_vectorized(self) -> np.ndarray:
        """
        The single variable Gausian liklihood function
        :param measured: The measured variable (NxC)
        :param estimated: The estimated variable or calculated from a model (NxC)
        :param N: An integer indicating the number of measured samples
        :param C: An integer indicating the number of Chains
        :param sigma: The standard deviation of the error estimation (1xC)
        :return: A numpy array indicating the log_liklihood function (1xC)
        """

        vectorized_error = ((self.measured - self.estimated)**2).sum(axis = 0)
        log_gauss = - self.N*np.log(self.sigma * np.sqrt(2 * np.pi)) - (vectorized_error / (2 * self.sigma ** 2))
        return log_gauss
        #====================================================================================

    def gaussian_liklihood_multivariable(self) -> np.ndarray:
        """
        The log liklihood of the Multivariable gaussian distribution used for multivariable fitting (multivariables objective function)
        :param measured: KxN measured parameters (K dimentional parameters and N sampling points)
        :param estimated:KxN estimated parameters (K dimentional parameters and N sampling points)
        :param N: An integer indicating the number of measurements
        :param Covariance: A positive definite square matrix indicating the covariance matrix of the multivariable Normal distribution (KxK)
        :param K: The dimention of the multivariable gaussian distribution
        :return: The log liklihood of
        """

        if self.Diagonal:
            indexes = np.arange(self.K, dtype = int)
            inv_cov = self.Covariance.copy()
            inv_cov[indexes, indexes] = 1 / inv_cov[indexes, indexes]
            det_cov = self.Covariance.diagonal().prod()
        else:
            inv_cov = np.linalg.inv(self.Covariance)     # calcualting the inversion of the covariance matrix
            det_cov = np.linalg.det(self.Covariance)     # calcualting the determinent of the covariance matrix

        Error = self.measured - self.estimated                 # KxN error matrix
        log_liklihood_gaussian = -self.N * np.log(np.sqrt(((2 * np.pi)**self.K) * det_cov)) - (0.5 * (np.diag(Error.T @ inv_cov @ Error))).sum()  # the log_liklihood gaussian
        return log_liklihood_gaussian


    def gaussian_liklihood_multivariable_vectorized(self)->np.ndarray:
        """
        The log liklihood of the Multivariable gaussian distribution used for multivariable fitting (multivariables objective function)
        :param measured: KxNxC measured parameters (K dimentional parameters and N sampling points and C chains)
        :param estimated:KxNxC estimated parameters (K dimentional parameters and N sampling points and C chains)
        :param N: An integer indicating the number of measurements
        :param Covariance: A positive definite square matrix indicating the covariance matrix of the multivariable Normal distribution (KxKxC)
        :param K: The dimention of the multivariable gaussian distribution
        :return: The log liklihood of the multivariable gaussian distribution
        """
        diagonal_indexes = np.arange(self.K, dtype=int)
        diagonal_indexes_samples = np.arange(self.N, dtype=int)
        inv_cov = self.Covariance.copy()   # predefining the inversion matrices
        Error = self.measured - self.estimated  # KxNxC error matrix
        Error_T = np.transpose(Error, axes = (1, 0, 2))  # NxKxC error matrix

        if self.Diagonal:
                # calcualting the inversion of the covariance matrix
            inv_cov[diagonal_indexes, diagonal_indexes, :] = 1 / inv_cov[diagonal_indexes, diagonal_indexes,:]  #KxKxX tensor
            det_cov = np.prod(self.Covariance[diagonal_indexes, Covariance,:], axis = 0)      # 1xC array

            vectorized_mahalanobis_distance = (((Error_T[:, :, None] * inv_cov).sum(axis = 1))[:, :, None] * T).sum(axis = 1) # NxNxC
            mahalanobis_distance = vectorized_mahalanobis_distance[diagonal_indexes_samples, diagonal_indexes_samples, :].sum(axis = 0)
            log_liklihood = -self.N * np.log(np.sqrt(((2 * np.pi) ** self.K) * det_cov)) - 0.5 * mahalanobis_distance
            return log_liklihood

        elif not Diagonal:
            det_cov = np.zeros((1,C))
            for c in range(C):
                det_cov[0, c] = np.linalg.det(self.Covariance[:, :, c])
                inv_cov[:, :, c:c+1] = np.linalg.inv(self.Covariance[:, :, c])

            vectorized_mahalanobis_distance = (((Error_T[:, :, None] * inv_cov).sum(axis=1))[:, :, None] * T).sum(axis = 1)  # NxNxC
            mahalanobis_distance = vectorized_mahalanobis_distance[diagonal_indexes_samples, diagonal_indexes_samples, :].sum(axis = 0)
            log_liklihood = -self.N * np.log(np.sqrt(((2 * np.pi) ** self.K) * det_cov)) - 0.5 * mahalanobis_distance
            return log_liklihood
        else:
            raise Exception('The type of calculating the liklihood function is not correctly specified!')