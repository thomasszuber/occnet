# %% Import modules 
import numpy as np

import numpy.random as rd 

# %% Define Occupational network class 

def RandomGraph(n,alpha=0.1,seed=123):
    rd.seed(seed)
    return (rd.rand(n,n) < alpha)*1




class OccNet:
    """
        Define an occupational network class which stores relevant information (parameters, wage, tightness...)

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
        self.w = np.ones(self.n) 
        self.thetas = rd.rand(self.n) 
        self.alpha = alpha 
        self.p = lambda x: x**alpha

# Find wages consistent with the free entry conditions
 
    def get_FE_wages(self):
        self.w = self.y - (self.r + self.s)*self.c*self.thetas/self.p(self.thetas)

# Method to solve for U: needs some more work...

    def get_U(self):
        self.U = np.ones(self.n)

# Find wages consistent with the Nash bargaining condition  

    def get_NB_wages(self): 
        return (1-self.phi)*self.y+self.phi*self.r*self.U

# Method to get FE/NB wage gap 

    def get_wage_gap(self): 
        return np.sum((self.w-self.get_NB_wages())**2)

