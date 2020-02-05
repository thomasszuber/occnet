# %% Import modules 
import numpy as np

from numpy import linalg as la 

import numpy.random as rd 

from scipy.optimize import minimize 

# %% Define Occupational network class 

def RandomGraph(n,alpha=0.1,seed=123):
    rd.seed(seed)
    G = (rd.rand(n,n) < alpha/2)*1
    G = ((G+np.transpose(G))>0)*1
    np.fill_diagonal(G,1)
    return G

def RandomDiGraph(n,alpha=0.1,seed=123):
    rd.seed(seed)
    G = (rd.rand(n,n) < alpha)*1
    np.fill_diagonal(G,1)
    return G 

# %%

class Method: 
    def __init__(self,opt='Nelder-Mead'):
        self.opt = opt 


class OccNet:
    """
        Define an occupational network class which stores relevant information (parameters,
         wage, tightness...)

    """ 
    def __init__(self,
                G,
                y=1,
                b=0.5,
                phi=0.5,
                r=0.01,
                s=0.03,
                c=0.1,
                alpha = 0.5):
        self.G = G
        self.n = G.shape[0]
        self.y = y 
        self.b = b
        self.phi = phi
        self.c = c
        self.r = r
        self.s = s
        self.alpha = alpha
        self.thetas = rd.rand(self.n)

# Update thetas: 

    def update_thetas(self,thetas):
        self.thetas = thetas 

    def get_p(self):
        self.p = self.thetas**self.alpha

# Find wages consistent with the free entry conditions
 
    def get_FE_wages(self):
        self.w = self.y - (self.r + self.s)*self.c*self.thetas/self.p

# Define optimal search strategy (matrix S)

    def get_S(self): 
        self.S = np.sum(self.G,axis=1)
        self.S = self.G/self.S[:,None]
        
# Define continuous time transition matrix 
    
    def get_PI(self): 
        self.PI = np.zeros((2*self.n,2*self.n))
        self.PI[0:self.n,self.n:2*self.n] = self.S*self.p
        self.PI[self.n:2*self.n,0:self.n] = np.diag([self.s for i in range(self.n)])
        np.fill_diagonal(self.PI,-np.sum(self.PI,axis=1))
        
    def get_PI_discrete(self): 
        self.PI = np.zeros((2*self.n,2*self.n))
        self.PI[0:self.n,self.n:2*self.n] = self.S*self.p
        self.PI[self.n:2*self.n,0:self.n] = np.diag([self.s for i in range(self.n)])
        self.PI = self.PI/np.sum(self.PI,axis=1)[:,None]
    
    def get_pi(self):
        self.get_PI()
        return la.eig(np.transpose(self.PI))
        
    def get_pi_discrete(self):
        self.get_PI_discrete()
        return la.eig(np.transpose(self.PI))
        
# Method to solve for U 

    def get_U(self):
        A = np.diag(self.r + np.sum(self.S*self.p,axis=1))
        B = self.b + np.sum(self.S*self.p*self.w,axis=1)/(self.r+self.s)
        C = self.S*self.p*self.s/(self.r+self.s)
        self.U = np.linalg.solve(A-C,B)

    def get_E(self): 
        self.E = (self.w+self.s*self.U)/(self.s+self.r)

    def check_S(self,update=False):
        self.get_E()
        G = ((np.repeat([self.E],self.n,axis=0)-self.U[:,None]) >= 0.0)*self.G
        S = np.sum(G,axis=1)
        S = G/S[:,None]
        if update == True:
            if not (S == self.S).all():
                self.S = S
                print("Updated search strategy.")
        else:
            return (S == self.S).all()
        

# Find wages consistent with the Nash bargaining condition  

    def get_NB_wages(self): 
        return self.phi*self.y+(1-self.phi)*self.r*self.U

# Method to get FE/NB wage gap 

    def get_wage_gap(self,thetas):
        self.update_thetas(thetas)
        self.get_p()
        self.get_FE_wages()
        self.get_S()
        self.get_U()
        return np.sum((self.w-self.get_NB_wages())**2)

    def get_equilibrium_thetas(self,init_point=None,method=Method()):
        if init_point == None: 
            init_point = np.ones(self.n)
        self.res = minimize(lambda x: self.get_wage_gap(x),init_point,method=method.opt)
        if self.res.success == True:
            print('Equilibrium found')
            self.update_thetas(self.res.x)
            self.get_p()
            self.get_FE_wages()


# Method to get equilibrium unemployment: 

    def get_u(self):
        A = np.dot(np.diag(self.p/np.sum(self.S*self.p,axis=1)),np.transpose(self.S))
        return la.eig(A)





