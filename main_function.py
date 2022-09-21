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
        ## loss function for minimization
        P = np.eye(self.L)
        F = x.T@P@x
        ## linear constraints
        R = self.A@self.x - self.b  # =0
        ## The Lagrangian for problem
        L = F + self.y.T@R
        return L



    def Dual_Ascent(self, A: np.ndarray = np.eye(1), b: np.ndarray = np.eye(1), alpha :np.float=0.1):

        """
        minimize f(x)
        subject to Ax=b
        with variable x ∈ Rn, where A ∈ Rm×n and f : Rn → R is convex
        The Lagrangian for the problem is L(x,y) = f(x) + yT (Ax − b)
        and the dual function is g(y) = inf x L(x,y) = −f ∗(−AT y) − bT y
        :param A: the matrix of constraint ∈ R(mxn) m linear equation and n adjustable parameters
        :param b:
        :param alpha: the learning rate
        :return:
        """

        m,n = A.shape
        m2,n2 = b.shape
        if m != m2:
            raise Exception('The number of parameters and equation is not consistent!')
        if n2 != 1:
            raise Exception('Currently the algorithms is not suitable for multi-output problems!')


        x = np.random.randn(self.L,1)
        y =  np.random.randn(1,n1)
        self.y = y
        self.x = y
        self.A = A
        self.b = b


