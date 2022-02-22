import numpy as np
from rpy2 import robjects as ro
r = ro.r
r['source']('install_lib.R')
