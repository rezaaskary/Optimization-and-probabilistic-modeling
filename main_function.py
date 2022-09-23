import numpy as np
import scipy as sc
from tqdm import tqdm



class optimizer:
    def __init__(self, algorithm: str='sgd', epsilon: float = None, beta1 :float = None, beta2 :float = None):
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.algorithm = algorithm

    def gradient_desent(self, parameter):






class Convex_problems:
    def __init__(self,problem_type: int=1, L:int = 1):
        """
        :param problem_type:
        :param L: the number of parameter for the ajustmnet
        """
        self.problem_type = problem_type
        self.L = L
        self.old_opt = np.inf
        self.parameter_optimization = (1e65) * np.ones((self.L, 1))
    # =================================================================
    def loss_f(self,x):
        self.P = np.eye(len(x))
        F = x.T @self.P @ x
        return F.ravel()
    #=======================================================================
    def linear_constraint(self,x):
        R = self.A @ x - self.b
        return R.ravel()
    #============================================================
    def lagrangian(self,x,y):
        """
        This function returns the value and the
        :param x:
        :param y:
        :return:
        """
        L = self.loss_f(x) + y.T @ self.linear_constraint(x)
        dL_dx = (self.P + self.P.T)@self.x + self.A.T@self.x
        dL_dy = self.A @ self.x - self.b
        return L.ravel(), dL_dx, dL_dy
    #=================================================================
    def dual_ascent_problem(self):
        """
        Calculating the partial derivativs with respect to the variables and the lagrange multiplier
        :return:
        """
        temp_x = self.x
        temp_y = self.y

        self.dL_dx = np.zeros((self.L,1))
        self.dL_dy = np.zeros((self.m, 1))
        self.h = 1e-12

        if self.derivatives_method == 'quadratic':
            # calcualting the derivatives with respect to x
            for i in range(self.L):
                x_r,x_l = self.x.copy(),self.x.copy()
                x_r[i] += self.h
                x_l[i] -= self.h
                self.dL_dx[i,0] = (1/(2*self.h))*(self.lagrangian(x_r,self.y) - self.lagrangian(x_l,self.y))
            # calcualting the derivatives with respect to y
            for i in range(self.m):
                y_r, y_l = self.y.copy(), self.y.copy()
                y_r[i] += self.h
                y_l[i] -= self.h
                self.dL_dy[i,0] = (1/(2*self.h))*(self.lagrangian(self.x, y_r) - self.lagrangian(self.x, y_l))

        elif self.derivatives_method == 'quartic':     # 4th oder numerical derivatives
            # calcualting the derivatives with respect to x
            for i in range(self.L):
                x_rr,x_ll,x_r,x_l = self.x.copy(),self.x.copy(),self.x.copy(),self.x.copy()
                x_rr[i] += 2*self.h
                x_r[i] += self.h
                x_ll[i] -= 2*self.h
                x_l[i] -= self.h
                self.dL_dx[i,0] = (1/(12*self.h))*(-self.lagrangian(x_rr, self.y) + 8.0*self.lagrangian(x_r, self.y)\
                                         - 8.0*self.lagrangian(x_l, self.y) + self.lagrangian(x_ll, self.y))
            # calcualting the derivatives with respect to y
            for i in range(self.m):
                y_rr, y_ll, y_r, y_l = self.y.copy(), self.y.copy(), self.y.copy(), self.y.copy()
                y_rr[i] += 2 * self.h
                y_r[i] += self.h
                y_ll[i] -= 2 * self.h
                y_l[i] -= self.h
                self.dL_dy[i, 0] = (1 / (12 * self.h)) * (-self.lagrangian(self.x, y_rr) + 8.0 * self.lagrangian(self.x, y_r) \
                            - 8.0 * self.lagrangian(self.x, y_l) + self.lagrangian(self.x, y_ll))
        else:
            raise Exception('Select a proper numerical method for the calculation of the first derivatives!')
        self.Lag = self.lagrangian(self.x,self.y)
        self.opt = self.loss_f(self.x)

    #===========================================================================
    def Dual_Ascent(self, A: np.ndarray = np.eye(1), b: np.ndarray = np.eye(1), alpha :float=0.1, tolerance: float=1e-6):
        """
        minimize f(x)
        subject to Ax=b
        with variable x ∈ Rn, where A ∈ Rm×n and f : Rn → R is convex
        The Lagrangian for the problem is L(x,y) = f(x) + yT (Ax − b)
        and the dual function is g(y) = inf x L(x,y) = −f ∗(−AT y) − bT y
        :param A:
        :param b:
        :param alpha:
        :param tolerance:
        :return:
        """

        self.tolerance = tolerance
        self.alpha = 0.01;
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

        self.y = np.random.randn(m,1)
        self.x =  np.random.randn(n,1)
        self.A = A
        self.b = b
        self.m = m
        self.iterations = 5000
        self.derivatives_method = 'quadratic'
        self.derivatives_method = 'quartic'

        # solving by using gradient decent approach
        for itr in tqdm(range(self.iterations)):
            self.dual_ascent_problem()
            self.x = self.x - self.alpha*self.dL_dx
            # self.y = self.y + self.alpha*(self.A@self.x-self.b)
            self.y = self.y + self.alpha *self.dL_dy
            # error = (np.abs(self.old_opt - self.opt)).sum()
            if self.opt<self.old_opt:
                error = (np.abs(self.old_opt - self.opt)).sum()
                self.old_opt = self.opt
                self.parameter_optimization = self.x
                if error<self.tolerance:
                    print('minimum realtive error achieved!')
                    break
        print(f'norm = :  {((self.A @ self.x - self.b) ** 2).sum()}')
        if itr == self.iterations-1:
            print('Optimization terminated due to the maximum iteration!')
        return self.x, self.opt, np.abs(self.opt - self.old_opt)/self.opt

        print()








if __name__=='__main__':
    A = np.random.rand(3,4)
    b = np.random.rand(3,1)
    D = Convex_problems(problem_type = 1, L= 4)
    D.Dual_Ascent(A=A,b=b,alpha=0.01)


