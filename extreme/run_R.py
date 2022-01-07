import numpy as np
from rpy2 import robjects as ro
import rpy2.robjects.numpy2ri


rpy2.robjects.numpy2ri.activate()
# Defining the R script and loading the instance in Python
r = ro.r
r['source']('revt.R')
# # Loading the function we have defined in R.
# test_func = ro.globalenv['test_func']
test_func = r.get_p_k

x = np.array([1.,2.,3.])
gamma=0.5
k_anchor=[2, 3, 4]
p, k = test_func(x, gamma, k_anchor)
print(p, k)