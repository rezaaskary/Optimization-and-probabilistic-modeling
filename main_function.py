import numpy as np
import scipy as sc


def



class Convex_problems:
    def __init__(self,problem_type: int=1):
        self.problem_type = problem_type





    def Dual_Ascent_model(self,x):
        """
        define a function f : Rn → R
        :param x:       R nx1
        :return:
        """




    def Dual_Ascent(self,x: np.ndarray = np.eye(1),y: np.ndarray = np.eye(1)):
        """
        minimize f(x)
        subject to ax=b
        with variable x ∈ Rn, where A ∈ Rm×n and f : Rn → R is convex
        The Lagrangian for the problem is L(x,y) = f(x) + yT (Ax − b)
        and the dual function is g(y) = inf x L(x,y) = −f ∗(−AT y) − bT y
        :return:
        """
        n1,m = x.shape
        n2,p = y.shape
        if n1 < m:
            raise Exception('The number of parameters is higher than observations!')
        if n1!=n2:
            raise Exception('the input and the output should have the same number of observations!')
