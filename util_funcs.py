import numpy as np
import pandas as pd
import seaborn as sns
import scipy
import matplotlib.pyplot as plt
import os
import sys
from counterfactual_tpp import sample_counterfactual, superposition, combine, check_monotonicity, distance
from sampling_utils import homogenous_poisson, thinning, thinning_T
from multiprocessing import cpu_count, Pool
from tqdm import tqdm
import utils

def normal(x, mean, sd, amp):  
    return amp * (1/(sd * (np.sqrt(2*np.pi)))) * np.exp(-0.5*((x-mean)/sd)**2)



