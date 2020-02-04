# %% Set directories 

import os
cwd = os.getcwd()
data = os.path.join(cwd,'../data')
results = os.path.join(cwd,'../results')
tmp = os.path.join(cwd,'../tmp')

# %% Import modules

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt 

import numpy.random as rd 

import networkx as nx

from scipy.optimize import minimize 


# %% Classes  
from classes import OccNet, RandomGraph, Method, RandomDiGraph

# %% Play with toys 

n = 10 

G = RandomDiGraph(n,alpha=0.3,ergodic=True)

# %%

self = OccNet(G,y=rd.chisquare(2,n),b=0.1,s=0.3)

self.get_graph_properties()

# %%

self.get_equilibrium_thetas(method=Method(opt='bfgs'))

self.check_S()




# %%

pi2 = self.get_pi_discrete() # erreur quelque part dans get_pi_discrete 

