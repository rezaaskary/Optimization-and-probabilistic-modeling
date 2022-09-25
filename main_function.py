import numpy as np
import scipy as sc
from tqdm import tqdm


#=================================================================================
class Optimizer:
    def __init__(self, algorithm: str='SGD',alpha: float = 0.2,\
                 epsilon: float = None, beta1 :float = None, type_of_optimization :str ='min',\
                 beta2 :float = None, dimention: int=1):
        self.epsilon_adam = epsilon
        self.beta1_adam = beta1
        self.beta2_adam = beta2
        self.algorithm = algorithm
        self.alpha = alpha
        self.dimention = dimention
        self.bias_vector = np.abs(np.random.randn(self.dimention, 1))
        self.m_adam =  np.abs(np.random.randn(self.dimention, 1))
        self.m_hat_adam =  np.abs(np.random.randn(self.dimention, 1))
        self.v_adam =  np.abs(np.random.randn(self.dimention, 1))
        self.v_hat_adam =  np.abs(np.random.randn(self.dimention, 1))
        if type_of_optimization == 'min':
            self.type_of_optimization = -1
        elif type_of_optimization == 'max':
            self.type_of_optimization = 1
        else:
            raise Exception('Please correctly enter the type of optimization!')

        if self.algorithm == 'SGD':
            self.fit = self.SGD
        elif self.algorithm == 'ADAM':
            self.fit = self.ADAM
        elif self.algorithm == 'RMSprop':
            self.fit = self.RMSprop
        else:
            raise Exception('Please use a correct optimizer')

    def SGD(self, parameter, derivatives, t):
        parameter = parameter + self.type_of_optimization * self.alpha * derivatives
        return parameter

    def ADAM(self,parameter, derivatives, t):
        self.m_adam = self.beta1_adam * self.m_adam + (1 - self.beta1_adam) * derivatives
        self.v_adam = self.beta2_adam * self.v_adam + (1 - self.beta2_adam) * derivatives**2
        self.m_hat_adam = self.m_adam / (1 - self.beta1_adam**(t+1))
        self.v_hat_adam = self.v_adam / (1 - self.beta2_adam**(t+1))
        parameter = parameter + self.type_of_optimization * self.alpha * self.m_hat_adam / (np.sqrt(self.v_hat_adam ) + self.epsilon_adam)
        return parameter

    def RMSprop(self,parameter, derivatives, t):
        self.m_adam = self.beta1_adam * self.m_adam + (1 - self.beta1_adam) * derivatives**2
        parameter = parameter + self.type_of_optimization * self.alpha * derivatives / (np.sqrt(self.m_adam) + self.epsilon_adam)
        return parameter
##================================================================================================
##================================================================================================
##================================================================================================
class Convex_problems_dual_ascend():
    def __init__(self,problem_type: int=1, learning_rate:float = 0.05, algorithm:str='SGD', tolerance: float=1e-12):
        self.tolerance = tolerance
        self.problem_type = problem_type
        self.old_opt = np.inf
        self.learning_rate = learning_rate
        self.algorithm = algorithm
    # =================================================================
    def loss_f(self):
        self.P = np.eye(self.n)
        F = self.x.T @self.P @ self.x
        dF_dx = (self.P + self.P.T)@ self.x
        return F, dF_dx
    #=======================================================================
    def linear_constraint(self):
        R = (self.A @ self.x - self.b)
        dR_dx = self.A.T
        aug = R.T @ R
        adug_dx = 2 * self.A.T @ R
        return R, dR_dx, aug, adug_dx
    #============================================================
    def lagrangian(self):
        self.opt, dF_dx = self.loss_f()
        lin_cons, dR_dx,_,_ = self.linear_constraint()

        L = self.opt + self.y.T @ lin_cons
        dL_dx = dF_dx + dR_dx @ self.y
        dL_dy = lin_cons
        return L, dL_dx, dL_dy

    # ============================================================
    def augmented_lagrangian(self):
        self.rho = 0.01
        self.opt, dF_dx = self.loss_f()
        lin_cons, dR_dx, aug, adug_dx = self.linear_constraint()

        L = self.opt + self.y.T @ lin_cons + (self.rho/2) * aug

        dL_dx = dF_dx + dR_dx @ self.y + (self.rho/2) * adug_dx
        dL_dy = lin_cons
        return L.ravel(), dL_dx, dL_dy
    #===========================================================================
    def Dual_Ascent(self, A: np.ndarray = np.eye(1), b: np.ndarray = np.eye(1), tolerance: float=1e-12):
        self.tolerance = tolerance
        m,n = A.shape       # m is the number of linear constraints
        m2,n2 = b.shape
        if m>n:
            raise Exception('Overdetermined Problem!')
        if m != m2:
            raise Exception('The number of parameters and equation is not consistent!')

        if n2 != 1:
            raise Exception('Currently the algorithms is not suitable for multi-output problems!')

        self.y = np.random.randn(m,1)
        self.x =  np.random.randn(n,1)
        self.A = A
        self.b = b
        self.m = m
        self.n = n
        self.iterations = 20000

        variable_optimizer = Optimizer(algorithm = self.algorithm, alpha = self.learning_rate, type_of_optimization = 'min')
        lagrange_optimizer = Optimizer(algorithm = self.algorithm, alpha = self.learning_rate, type_of_optimization = 'max')

        for itr in tqdm(range(self.iterations)):
            L, dl_dx, dl_dy = self.lagrangian()
            L, dl_dx, dl_dy = self.augmented_lagrangian()
            self.x = variable_optimizer.fit(self.x, dl_dx, itr//1000)
            self.y = lagrange_optimizer.fit(self.y, dl_dy, itr//1000)
            tol = np.abs(self.opt - self.old_opt)
            self.old_opt = self.opt
            if tol<self.tolerance:
                print('Optimum values acheived!')
                break

        if itr == self.iterations - 1:
            print('Optimization terminated due to the maximum iteration!')

        print(f'norm of constraint Error= :  {((self.A @ self.x - self.b) ** 2).sum()}')
        print(f'the value of loss function= :  {self.opt}')
        return self.x, self.opt
#===============================================================================================

# if __name__=='__main__':
#     A = np.random.rand(10,12)
#     b = np.random.rand(10,1)
#     D = Convex_problems_dual_ascend(problem_type = 1, learning_rate=0.05, algorithm='SGD')
#     val,opt = D.Dual_Ascent(A=A, b=b)
#     val



#=================================================================================
#=================================================================================
#=================================================================================
class ADMM:
    def __init__(self,problem_type: int=1, learning_rate:float = 0.05, algorithm:str='SGD',tolerance: float=1e-12, iterations:int = 20000):
        self.tolerance = tolerance
        self.problem_type = problem_type
        self.old_opt = np.inf
        self.learning_rate = learning_rate
        self.algorithm = algorithm
        self.iterations = iterations
    # =================================================================
    def loss_f(self):
        self.P1 = np.eye(self.n1)
        self.P2 = np.eye(self.n2)

        F1 = self.x.T @self.P1 @ self.x
        F2 = self.z.T @ self.P2 @ self.z

        dF_dx = (self.P1 + self.P1.T) @ self.x
        dF_dz = (self.P2 + self.P2.T) @ self.z

        F = F1 + F2

        return F, dF_dx, dF_dz
    #=======================================================================
    def linear_constraint(self):

        R = self.A @ self.x + self.B @ self.z - self.c
        aug = R.T @ R

        dR_dx = self.A.T
        dR_dz = self.B.T

        adug_dx = 2 * dR_dx @ R
        adug_dz = 2 * dR_dz @ R

        return R, aug, dR_dx, dR_dz, adug_dx, adug_dz
    #============================================================
    def augmented_lagrangian(self):
        self.rho = 0.01
        F, dF_dx, dF_dz = self.loss_f()
        R, aug, dR_dx, dR_dz, adug_dx, adug_dz = self.linear_constraint()

        Cons = self.y.T @ R
        L = F + Cons + (self.rho/2) * aug

        dL_dx = dF_dx + dR_dx @ self.y + (self.rho/2) * adug_dx
        dL_dz = dF_dz + dR_dz @ self.y + (self.rho/2) * adug_dz
        dL_dy = R
        self.opt = F + (self.rho/2) * aug
        return L, dL_dx, dL_dz, dL_dy
    #===========================================================================
    def ADMM_dual_ascent(self, A: np.ndarray = np.eye(1), B: np.ndarray = np.eye(1), c: np.ndarray = np.eye(1)):

        p1,n1 = A.shape
        p2, n2 = B.shape
        p3,n3 = c.shape

        if p1==p2==p3:
            self.p = p1
        else:
            raise Exception('The matrices of linear constraint are not consistent!')

        self.n1 = n1
        self.n2 = n2

        self.y = np.random.randn(self.p,1)
        self.x =  np.random.randn(n1,1)
        self.z = np.random.randn(n2, 1)

        self.A = A
        self.B = B
        self.c = c

        variable_optimizer_x = Optimizer(algorithm = self.algorithm, alpha = self.learning_rate, type_of_optimization = 'min')
        variable_optimizer_z = Optimizer(algorithm = self.algorithm, alpha = self.learning_rate, type_of_optimization = 'min')
        lagrange_optimizer = Optimizer(algorithm = self.algorithm, alpha = self.learning_rate, type_of_optimization = 'max')

        for itr in tqdm(range(self.iterations)):
            L, dl_dx, dl_dz, dl_dy = self.augmented_lagrangian()

            self.x = variable_optimizer_x.fit(self.x, dl_dx, itr//1000)
            self.z = variable_optimizer_z.fit(self.z, dl_dz, itr // 1000)
            self.y = lagrange_optimizer.fit(self.y, dl_dy, itr//1000)

            tol = np.abs(self.opt - self.old_opt)
            self.old_opt = self.opt
            if tol<self.tolerance:
                print('Optimum values acheived!')
                break

        if itr == self.iterations - 1:
            print('Optimization terminated due to the maximum iteration!')

        print(f'norm of constraint Error= :  {((self.A @ self.x +self.B @ self.z - self.c) ** 2).sum()}')
        print(f'the value of loss function= :  {self.opt}')
        return self.x, self.opt


# if __name__=='__main__':
#     A = np.random.rand(10,12)
#     B = np.random.rand(10, 4)
#     c = np.random.rand(10,1)
#     D = ADMM(problem_type = 1, learning_rate=0.05, algorithm='SGD')
#     val,opt = D.ADMM_dual_ascent(A=A,B=B, c=c)
#     val


#=================================================================================
#=================================================================================
#=================================================================================
class ADMM:
    def __init__(self,problem_type: int=1, learning_rate:float = 0.05, algorithm:str='SGD',tolerance: float=1e-12, iterations:int = 20000):
        self.tolerance = tolerance
        self.problem_type = problem_type
        self.old_opt = np.inf
        self.learning_rate = learning_rate
        self.algorithm = algorithm
        self.iterations = iterations
    # =================================================================
    def loss_f(self):
        self.P1 = np.eye(self.n1)
        self.P2 = np.eye(self.n2)

        F1 = self.x.T @self.P1 @ self.x
        F2 = self.z.T @ self.P2 @ self.z

        dF_dx = (self.P1 + self.P1.T) @ self.x
        dF_dz = (self.P2 + self.P2.T) @ self.z

        F = F1 + F2

        return F, dF_dx, dF_dz
    #=======================================================================
    def linear_constraint(self):

        R = self.A @ self.x + self.B @ self.z - self.c
        aug = R.T @ R

        dR_dx = self.A.T
        dR_dz = self.B.T

        adug_dx = 2 * dR_dx @ R
        adug_dz = 2 * dR_dz @ R

        return R, aug, dR_dx, dR_dz, adug_dx, adug_dz
    #============================================================
    def augmented_lagrangian(self):
        self.rho = 0.01
        F, dF_dx, dF_dz = self.loss_f()
        R, aug, dR_dx, dR_dz, adug_dx, adug_dz = self.linear_constraint()

        Cons = self.y.T @ R
        L = F + Cons + (self.rho/2) * aug

        dL_dx = dF_dx + dR_dx @ self.y + (self.rho/2) * adug_dx
        dL_dz = dF_dz + dR_dz @ self.y + (self.rho/2) * adug_dz
        dL_dy = R
        self.opt = F + (self.rho/2) * aug
        return L, dL_dx, dL_dz, dL_dy
    #===========================================================================
    def ADMM_dual_ascent(self, A: np.ndarray = np.eye(1), B: np.ndarray = np.eye(1), c: np.ndarray = np.eye(1)):

        p1,n1 = A.shape
        p2, n2 = B.shape
        p3,n3 = c.shape

        if p1==p2==p3:
            self.p = p1
        else:
            raise Exception('The matrices of linear constraint are not consistent!')

        self.n1 = n1
        self.n2 = n2

        self.y = np.random.randn(self.p,1)
        self.x =  np.random.randn(n1,1)
        self.z = np.random.randn(n2, 1)

        self.A = A
        self.B = B
        self.c = c

        variable_optimizer_x = Optimizer(algorithm = self.algorithm, alpha = self.learning_rate, type_of_optimization = 'min')
        variable_optimizer_z = Optimizer(algorithm = self.algorithm, alpha = self.learning_rate, type_of_optimization = 'min')
        lagrange_optimizer = Optimizer(algorithm = self.algorithm, alpha = self.learning_rate, type_of_optimization = 'max')

        for itr in tqdm(range(self.iterations)):
            L, dl_dx, dl_dz, dl_dy = self.augmented_lagrangian()

            self.x = variable_optimizer_x.fit(self.x, dl_dx, itr//1000)
            self.z = variable_optimizer_z.fit(self.z, dl_dz, itr // 1000)
            self.y = lagrange_optimizer.fit(self.y, dl_dy, itr//1000)

            tol = np.abs(self.opt - self.old_opt)
            self.old_opt = self.opt
            if tol<self.tolerance:
                print('Optimum values acheived!')
                break

        if itr == self.iterations - 1:
            print('Optimization terminated due to the maximum iteration!')

        print(f'norm of constraint Error= :  {((self.A @ self.x +self.B @ self.z - self.c) ** 2).sum()}')
        print(f'the value of loss function= :  {self.opt}')
        return self.x, self.opt



