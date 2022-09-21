import numpy as np
import scipy as sc



class Convex_problems:
    def __init__(self,problem_type: int=1, L:int = 1):
        """
        :param problem_type:
        :param L: the number of parameter for the ajustmnet
        """
        self.problem_type = problem_type
        self.L = L

    def Dual_Ascent_problem(self):
        """
        In this method, you should define the  loss function F and its first derivatves
        define a function f : Rn → R
        :param x:       R mx1
        :return:
        """
        # =======================================================================
        ## loss function (a quadratic function is used as an example)
        x = self.x
        A = self.A
        b = self.b
        def Loss_F(x):
            P = np.eye(len(x))
            F = x.T @ P @ x
            return F

        def Linear_Constraint(A,b,x):
            R = self.A @ self.x - self.b
            return R
        #=======================================================================
        dL_dx = np.zeros((self.L,1))
        h = 1e-12
        for i in range(self.L):
            x_r,x_l = x,x
            x_r[i] += h
            x_l[i] -= h
            dL_dx[i,0] = (1/(2*h))*(Loss_F(x_r)-Loss_F(x_l)) +\
                         (1/(2*h))*(Linear_Constraint(A,b,x_r) - Linear_Constraint(A,b,x_l))

        F = Loss_F(x)
        R = Linear_Constraint(A,b,x)
        L = F + self.y.T@R
        return L, dL_dx



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
        alpha = 0.01;
        m,n = A.shape       # m is the number of linear constraints
        m2,n2 = b.shape

        if m != m2:
            raise Exception('The number of parameters and equation is not consistent!')

        if n2 != 1:
            raise Exception('Currently the algorithms is not suitable for multi-output problems!')

        if self.L != n:
            raise Exception('the dimention of variables and the problem is not consistent!')

        x = np.random.randn(n,1)
        y =  np.random.randn(m,1)
        self.y = y
        self.x = y
        self.A = A
        self.b = b

        iterations = 100;
        for itr in range(iterations):
            Q = self.Dual_Ascent_problem()





        return 1

