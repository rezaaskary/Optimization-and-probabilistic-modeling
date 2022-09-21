import numpy as np
import scipy as sc


def loss_function



class Convex_problems:
    def __init__(self,problem_type: int=1, L:int = 1):
        """

        :param problem_type:
        :param L: the number of parameter for the ajustmnet
        """
        self.problem_type = problem_type
        self.L = L

    def Dual_Ascent_model(self):
        """
        define a function f : Rn → R
        :param x:       R mx1
        :return:
        """
        P = np.eye(self.L)
        F = x.T@P@x
        L = F +



    def Dual_Ascent(self, A: np.ndarray = np.eye(1), b: np.ndarray = np.eye(1), alpha :np.float=0.1):
        """
        minimize f(x)
        subject to Ax=b
        with variable x ∈ Rn, where A ∈ Rm×n and f : Rn → R is convex
        The Lagrangian for the problem is L(x,y) = f(x) + yT (Ax − b)
        and the dual function is g(y) = inf x L(x,y) = −f ∗(−AT y) − bT y
        :return:
        """
        n1,m = A.shape
        n2,p = b.shape
        if n1 < m:
            raise Exception('The number of parameters is higher than observations!')
        if n1!=n2:
            raise Exception('the input and the output should have the same number of observations!')
        if p!=1:
            raise Exception('Currently the algorithms is not suitable for multi-output problems!')
        if self.L != m:
            raise Exception('The number of adjustable parameters and the dimention of the data is not compatible!')

        x = np.random.randn(self.L,1)
        y =  np.random.randn(1,self.L)
        self.y = y
        self.x = y
        self.A = A
        self.b = b


