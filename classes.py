# %% Import modules 
import numpy as np

from numpy import linalg as la 

import numpy.random as rd 

import networkx as nx

from scipy.optimize import minimize 

# %% Define Occupational network class 

def RandomGraph(n,alpha=0.1,ergodic=True,seed=123):
    rd.seed(seed)
    G = (rd.rand(n,n) < alpha/2)*1
    np.fill_diagonal(G,1)
    if ergodic: 
        for i in range(n-1):
            G[i,i+1] = 1
        G[n-1,0] = 1
    G = ((G+np.transpose(G))>0)*1
    return G

def RandomDiGraph(n,alpha=0.1,ergodic=True,seed=123):
    rd.seed(seed)
    G = (rd.rand(n,n) < alpha)*1
    np.fill_diagonal(G,1)
    if ergodic: 
        for i in range(n-1):
            G[i,i+1] = 1
        G[n-1,0] = 1
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
        
        if type(y) != np.ndarray:
            self.y = np.repeat([y],G.shape[0])
        else: 
            self.y = y
        
        if type(b) != np.ndarray:
            self.b = np.repeat([b],G.shape[0])
        else: 
            self.b = b
        
        if type(phi) != np.ndarray:
            self.phi = np.repeat([phi],G.shape[0])
        else: 
            self.phi = phi

        self.c = c
        self.r = r
        self.s = s
        self.alpha = alpha
        
# Update graph properties 
    
    def get_graph_properties(self): 
        if (self.G == self.G.T).all():
            self.g = nx.Graph(self.G)
        else: 
            self.g = nx.DiGraph(self.G)
        self.eig = nx.eigenvector_centrality(self.g)
        self.katz = nx.katz_centrality(self.g)
        self.degree = nx.degree_centrality(self.g)
        

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
        
# Define continuous time transition matrix: 
    
    def get_PI(self):
        """
            The transition matrix is block defined as: 
            PI =  [[U,U],[U,E],
                 [[E,U],[E,E]]
            Where PI[i,j] is the transition probability from state i to state j  
        """
        self.PI = np.zeros((2*self.n,2*self.n))
        self.PI[0:self.n,self.n:2*self.n] = self.S*self.p # [U,E]
        self.PI[self.n:2*self.n,0:self.n] = np.diag([self.s for i in range(self.n)]) # [E,U]
        np.fill_diagonal(self.PI,-np.sum(self.PI,axis=1)) # WITH NEGATIVE DIAGONAL EQUAL TO SUM OF LINE 
        
    
    def get_pi(self,tol=1e-12):
        self.get_PI()
        eig, vec = la.eig(np.transpose(self.PI))
        eig = np.abs(eig)
        zero = np.argmin(eig)
        if eig[zero] < tol:
            vec = vec[:,zero]
            self.pi = vec = vec/np.sum(vec,axis=0)
            self.pi_u = {i:x for i,x in enumerate(vec[0:self.n].real)}
            self.pi_e = {i:x for i,x in enumerate(vec[self.n::].real)}
            self.pi_l = {i:x+y for i,(x,y) in enumerate(zip(vec[0:self.n].real,vec[self.n::].real))}
            self.u = {i:x/y for i,(x,y) in enumerate(zip(self.pi_u.values(),self.pi_l.values()))}
        else: 
            print(f'0 not an eigenvalue at tol level {tol}')
            
    def get_PI_discrete(self): 
        self.PI = np.zeros((2*self.n,2*self.n))
        self.PI[0:self.n,self.n:2*self.n] = self.S*self.p
        self.PI[self.n:2*self.n,0:self.n] = np.diag([self.s for i in range(self.n)])
        self.PI = self.PI/np.sum(self.PI,axis=1)[:,None] # DIVIDE BY SUM OF LINE TO DISCRETIZE 
        
    def get_pi_discrete(self,tol=1e-12):
        self.get_PI_discrete()
        eig, vec = la.eig(np.transpose(self.PI))
        eig = np.abs(eig-1)
        zero = np.argmin(eig)
        if eig[zero] < tol:
            vec = vec[:,zero]
            return vec/np.sum(vec,axis=0)
        else: 
            print(f'1 not an eigenvalue at tol level {tol}')
        
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
            self.get_pi()
        else:
            if method.opt != 'Nelder-Mead':
                print(f'Switched from {method.opt} to Nelder-Mead')
                self.res = minimize(lambda x: self.get_wage_gap(x),init_point,method='Nelder-Mead')
                if self.res.success == True:
                    print('Equilibrium found')
                    self.update_thetas(self.res.x)
                    self.get_p()
                    self.get_FE_wages()
                    self.get_pi()
                else: 
                    print('No equilibrium found.')






