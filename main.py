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

G = RandomDiGraph(30,alpha=0.1)

new = OccNet(G)

new.get_FE_wages()

new.get_S()

new.get_U()

res = new.get_equilibrium_thetas()





# %%

g = nx.DiGraph(G)


# %%
