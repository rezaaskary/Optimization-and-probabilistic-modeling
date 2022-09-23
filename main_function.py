import numpy as np
import scipy as sc
from tqdm import tqdm


class Convex_problems:
    def __init__(self,problem_type: int=1, L:int = 1):
        """
        :param problem_type:
        :param L: the number of parameter for the ajustmnet
        """
        self.problem_type = problem_type
        self.L = L

    # =================================================================
    def loss_f(self,x):
        P = np.eye(len(x))
        F = x.T @ P @ x
        return F.ravel()
    #=======================================================================
    def linear_constraint(self,x):
        R = self.A @ x - self.b
        return R.ravel()
    #============================================================
    def lagrangian(self,x,y):
        L = self.loss_f(x) + y.T @ self.linear_constraint(x)
        return L.ravel()
    #=================================================================
    def dual_ascent_problem(self):
        self.dL_dx = np.zeros((self.L,1))
        self.dL_dy = np.zeros((self.m, 1))
        self.h = 1e-12

        if self.derivatives_method == 'quadratic':
            # calcualting the derivatives with respect to x
            for i in range(self.L):
                x_r,x_l = self.x.copy(),self.x.copy()
                x_r[i] += self.h
                x_l[i] -= self.h
                self.dL_dx[i,0] = (1/(2*self.h))*(self.lagrangian(self.A,x_r,self.b,self.y) - self.lagrangian(self.A,x_l,self.b,self.y))
            # calcualting the derivatives with respect to y
            for i in range(self.m):
                y_r, y_l = self.y.copy(), self.y.copy()
                y_r[i] += self.h
                y_l[i] -= self.h
                self.dL_dy[i,0] = (1/(2*self.h))*(self.lagrangian(self.A,self.x,self.b,y_r) - self.lagrangian(self.A,self.x,self.b,y_l))

        elif self.derivatives_method == 'quartic ':     # 4th oder numerical derivatives
            # calcualting the derivatives with respect to x
            for i in range(self.L):
                x_rr,x_ll,x_r,x_l = self.x.copy(),self.x.copy(),self.x.copy(),self.x.copy()
                x_rr[i] += 2*self.h
                x_r[i] += self.h
                x_ll[i] -= 2*self.h
                x_l[i] -= self.h
                self.dL_dx[i,0] = (1/(12*self.h))*(-self.lagrangian(self.A,x_rr,self.b,self.y) + 8.0*self.lagrangian(self.A,x_r,self.b,self.y)\
                                         - 8.0*self.lagrangian(self.A,x_l,self.b,self.y) + self.lagrangian(self.A,x_ll,self.b,self.y))
            # calcualting the derivatives with respect to y
            for i in range(self.m):
                y_rr, y_ll, y_r, y_l = self.y.copy(), self.y.copy(), self.y.copy(), self.y.copy()
                y_rr[i] += 2 * self.h
                y_r[i] += self.h
                y_ll[i] -= 2 * self.h
                y_l[i] -= self.h
                self.dL_dy[i, 0] = (1 / (12 * self.h)) * (-self.lagrangian(self.A, self.x, self.b, y_rr) + 8.0 * self.lagrangian(self.A, self.x, self.b,y_r) \
                            - 8.0 * self.lagrangian(self.A, self.x, self.b, y_l) + self.lagrangian(self.A, self.x, self.b,y_ll))
        else:
            raise Exception('Select a proper numerical method for the calculation of the first derivatives!')
        self.Lag = self.lagrangian(self.A,self.x,self.b,self.y)
        self.opt = self.loss_f(self,x,A,b,y)

    #===========================================================================
    def Dual_Ascent(self, A: np.ndarray = np.eye(1), b: np.ndarray = np.eye(1), alpha :float=0.1, tolerance: float=1e-6):
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
        self.tolerance = tolerance
        self.alpha = 0.2;
        m,n = A.shape       # m is the number of linear constraints
        m2,n2 = b.shape

        if m>n:
            raise Exception('Overdetermined Problem!')
        if m != m2:
            raise Exception('The number of parameters and equation is not consistent!')

        if n2 != 1:
            raise Exception('Currently the algorithms is not suitable for multi-output problems!')

        if self.L != n:
            raise Exception('the dimention of variables and the problem is not consistent!')

        x = np.random.randn(n,1)
        y =  np.random.randn(m,1)
        self.y = y
        self.x = x
        self.A = A
        self.b = b
        self.m = m
        self.iterations = 2000
        self.derivatives_method = 'quadratic'

        for itr in tqdm(range(self.iterations)):
            self.dual_ascent_problem()
            # print(self.x.T@self.P@self.x)
            self.x = self.x - self.alpha*self.dL_dx
            self.y = self.y + self.alpha*(self.A@self.x-self.b)


            # print(f'norm = :  {((self.A@self.x-self.b)**2).sum()}')
                # print(f'lagrangian:  {L}')
        self.x







if __name__=='__main__':
    A = np.random.rand(4,4)
    b = np.random.rand(4,1)
    D = Convex_problems(problem_type = 1, L= 4)
    D.Dual_Ascent(A=A,b=b,alpha=0.01)


