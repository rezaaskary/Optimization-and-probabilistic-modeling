import numpy as np
import scipy as sc



class Convex_problems:
    def __init__(self,problem_type: int=1):
        self.problem_type = problem_type



    def Dual_Ascent(self):
        """
        minimize f(x)
        subject to ax=b
        with variable x ∈ Rn, where A ∈ Rm×n and f : Rn → R is convex
        The Lagrangian for the problem is L(x,y) = f(x) + yT (Ax − b)
        and the dual function is g(y) = inf x L(x,y) = −f ∗(−AT y) − bT y
        :return:
        """
