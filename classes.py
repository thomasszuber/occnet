# %% Import modules 
import numpy as np

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
        self.p = lambda x: x**alpha

# Find wages consistent with the free entry conditions
 
    def get_FE_wages(self):
        self.w = self.y - (self.r + self.s)*self.c*self.thetas/self.p(self.thetas)

# Define optimal search strategy (matrix S)

    def get_S(self): 
        self.S = np.sum(self.G,axis=1)
        self.S = self.G/self.S[:,None]

# Method to solve for U: needs some more work...

    def get_U(self):
        A = np.diag(self.r + np.sum(self.S*self.p(self.thetas),axis=1))
        B = self.b + np.sum(self.S*self.p(self.thetas)*self.w,axis=1)/(self.r+self.s)
        C = self.S*self.p(self.thetas)*self.s/(self.r+self.s)
        self.U = np.linalg.solve(A-C,B)

# Find wages consistent with the Nash bargaining condition  

    def get_NB_wages(self): 
        return (1-self.phi)*self.y+self.phi*self.r*self.U

# Method to get FE/NB wage gap 

    def get_wage_gap(self,thetas):
        self.thetas = thetas
        self.get_FE_wages()
        self.get_S()
        self.get_U()
        return np.sum((self.w-self.get_NB_wages())**2)

    def get_equilibrium_thetas(self,init_point=None,method=Method()):
        if init_point == None: 
            init_point = np.ones(self.n)
        return minimize(lambda x: self.get_wage_gap(x),init_point,method=method.opt)


