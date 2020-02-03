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


# %% Classes  
from classes import OccNet, RandomGraph

# %% Play with toys 

G = RandomGraph(100,alpha=0.1)

new = OccNet(G)

new.get_FE_wages()

new.get_U()

gap = new.get_wage_gap()

# %%

g = nx.DiGraph(G)


# %%
