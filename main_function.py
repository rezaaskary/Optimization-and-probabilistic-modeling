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
    def Linear_Constraint(self,A,x,b):
        R = A @ x - b
        return R.ravel()
    #============================================================
    def Lagrangian(self,A,x,b,y):
        L = self.Loss_F(x,A,b,y) + y.T @ self.Linear_Constraint(A, x, b)
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
        self.h = 1e-12
        # precision = 'quadratic'

        if self.derivatives_method == 'quadratic':
            for i in range(self.L):
                x_r,x_l = self.x.copy(),self.x.copy()
                x_r[i] += self.h
                x_l[i] -= self.h
                dL_dx[i,0] = (1/(2*self.h))*(self.Lagrangian(self.A,x_r,self.b,self.y) - self.Lagrangian(self.A,x_l,self.b,self.y))

            for i in range(self.m):
                y_r, y_l = self.y.copy(), self.y.copy()
                y_r[i] += self.h
                y_l[i] -= self.h
                dL_dy[i,0] = (1/(2*self.h))*(self.Lagrangian(self.A,self.x,self.b,y_r) - self.Lagrangian(self.A,self.x,self.b,y_l))


        elif precision == 'quartic ':
            #
            for i in range(self.L):
                x_rr,x_ll,x_r,x_l = self.x.copy(),self.x.copy(),self.x.copy(),self.x.copy()
                x_rr[i] += 2*self.h
                x_r[i] += self.h
                x_ll[i] -= 2*self.h
                x_l[i] -= self.h
                dL_dx[i,0] = (1/(12*self.h))*(-self.Lagrangian(self.A,x_rr,self.b,self.y) + 8.0*self.Lagrangian(self.A,x_r,self.b,self.y)\
                                         - 8.0*self.Lagrangian(self.A,x_l,self.b,self.y) + self.Lagrangian(self.A,x_ll,self.b,self.y))

            for i in range(self.m):
                y_rr, y_ll, y_r, y_l = self.y.copy(), self.y.copy(), self.y.copy(), self.y.copy()
                y_rr[i] += 2 * self.h
                y_r[i] += self.h
                y_ll[i] -= 2 * self.h
                y_l[i] -= self.h
                dL_dx[i, 0] = (1 / (12 * self.h)) * (-self.Lagrangian(self.A, x_rr, self.b, self.y) + 8.0 * self.Lagrangian(self.A, x_r, self.b,self.y) \
                            - 8.0 * self.Lagrangian(self.A, x_l, self.b, self.y) + self.Lagrangian(self.A, x_ll, self.b,self.y))



        # elif precision == 'analytical':
        #     return analytical_solution(A,y)
        # else:
        #     raise Exception('Select a proper numerical method for the calculation of the first derivatives!')

        L = self.Lagrangian(self.A,self.x,self.b,self.y)
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
        self.alpha = 0.2;
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
        self.x = x
        self.A = A
        self.b = b
        self.m = m
        self.iterations = 2000
        self.derivatives_method = 'quadratic'
        self.solution = 'numerical'
        if self.solution == 'numerical':
            xold = x.copy()
            yold = y.copy()
            for itr in range(self.iterations):
                L, dL_dx,dL_dy = self.Dual_Ascent_problem()
                self.P = np.eye(len(x))
                print(self.x.T@self.P@self.x)
                # dL_dx = (self.P+self.P.T)@self.x +self.A.T@self.y


                self.x = self.x - self.alpha*dL_dx
                self.y = self.y + self.alpha*(self.A@self.x-self.b)


                print(f'norm = :  {((self.A@self.x-self.b)**2).sum()}')
                # print(f'lagrangian:  {L}')
        self.x





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


