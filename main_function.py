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

    # =================================================================
    def Loss_F(self,x,A,b,y):
        P = np.eye(len(x))
        F = x.T @ P @ x
        return F.ravel()
    #=======================================================================
    def Linear_Constraint(self,x,A,b):
        R = A @ x - b
        return R.ravel()
    #============================================================
    def Lagrangian(self,A,x,b,y):
        L = Loss_F(x,A,b,y) + y.T @ Linear_Constraint(A, b, x)
        return L.ravel()
    #=============================================================
    def analytical_solution(self):
        """
        we solve the derivatives of dL/dx and obtain the value of x
        :param A:
        :param y:
        :return: x
        """
        return np.linalg.inv(self.A.T + self.A) @ self.A.T @ self.y
    #=================================================================
    def Dual_Ascent_problem(self):
        """
        In this method, you should define the  loss function F and its first derivatves
        define a function f : Rn → R
        :param x:       R mx1
        :return:
        L: lagrangian
        dL_dx: derivatives of lagrangian
        """

        #=======================================================================
        x = self.x
        y = self.y

        dL_dx = np.zeros((self.L,1))
        dL_dy = np.zeros((self.m, 1))
        h = 1e-8
        # precision = 'quadratic'

        if self.derivatives_method == 'quadratic':
            for i in range(self.L):
                x_r,x_l = x.copy(),x.copy()
                x_r[i] += h
                x_l[i] -= h
                dL_dx[i,0] = (1/(2*h))*(self.Lagrangian(A,x_r,b,y) - self.Lagrangian(A,x_l,b,y))

            for i in range(self.m):
                y_r, y_l = y.copy(), y.copy()
                y_r[i] += h
                y_l[i] -= h
                dL_dy[i,0] = (1/(2*h))*(Lagrangian(A,x,b,y_r) - Lagrangian(A,x,b,y_l))


        # elif precision == 'quartic ':
        #     for i in range(self.L):
        #         x_rr,x_ll,x_r,x_l = x.copy(),x.copy(),x.copy(),x.copy()
        #         x_rr[i] += 2*h
        #         x_r[i] += h
        #         x_ll[i] -= 2*h
        #         x_l[i] -= h
        #         dL_dx[i,0] = (1/(12*h))*(-Lagrangian(A,x_rr,b,y) + 8.0*Lagrangian(A,x_r,b,y) - 8.0*Lagrangian(A,x_l,b,y) + Lagrangian(A,x_ll,b,y))
        # elif precision == 'analytical':
        #     return analytical_solution(A,y)
        # else:
        #     raise Exception('Select a proper numerical method for the calculation of the first derivatives!')

        L = Lagrangian(A,x,b,y)
        return L, dL_dx,dL_dy
    #===========================================================================
    def Dual_Ascent(self, A: np.ndarray = np.eye(1), b: np.ndarray = np.eye(1), alpha :float=0.1):
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
        self.alpha = 0.01;
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
        self.m = m
        self.iterations = 1000
        self.derivatives_method = 'quadratic'
        self.solution = 'numerical'
        if self.solution == 'numerical':
            xold = x.copy()
            yold = y.copy()
            for itr in range(self.iterations):
                L, dL_dx,dL_dy = self.Dual_Ascent_problem()

                xnew = xold - self.alpha*dL_dx
                self.x = xnew
                ynew = yold - self.alpha*dL_dy
                self.y = ynew







        iterations = 1000;
        for itr in range(iterations):
            x_new = self.Dual_Ascent_problem()
            self.y = self.y + alpha*(self.A@x_new - self.b)
            self.x = x_new

        return self.x




if __name__=='__main__':
    A = np.random.rand(4,4)
    b = np.random.rand(4,1)
    D = Convex_problems(problem_type = 1, L= 4)
    D.Dual_Ascent(A=A,b=b,alpha=0.01)


