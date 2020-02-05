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

n = 7 

G = RandomDiGraph(n,alpha=0.3,ergodicity="ergodic")

# %%

#self = OccNet(G,y=rd.chisquare(2,n),b=0.1,s=0.3)

self = OccNet(G,y=1,b=0.1,s=0.3)

self.get_graph_properties()

# %%
method = Method(opt='Nelder-Mead',search_strategy="endogenous_effort")

self.get_equilibrium_thetas(method=method)

# %%

self.check_S(method=method)


# %%

# %%

pi2 = self.get_pi_discrete() # erreur quelque part dans get_pi_discrete 

