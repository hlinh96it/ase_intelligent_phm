import pandas as pd
import glob
import os
import argparse
import numpy as np
import ruptures as rpt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy import signal
import time
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

import warnings
warnings.filterwarnings("ignore")

plt.style.use('default')
plt.rcParams['axes.grid'] = True
plt.rcParams["figure.figsize"] = (16, 4)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['agg.path.chunksize'] = 100000


# %% Data reading
data_path_1 = r'/Users/hoanglinh96nl/Library/CloudStorage/OneDrive-NTHU/Projects and Papers/ASE_PHM_WireBonding/dataset/8_Nov15_test_crop/DAQ_20220324'
data_path_2 = r'/Users/hoanglinh96nl/Library/CloudStorage/OneDrive-NTHU/Projects and Papers/ASE_PHM_WireBonding/dataset/8_Nov15_test_crop/DAQ_20220328'

cell1_WBR431_20220316_004618_005623_0392k = os.path.join(data_path_1, os.listdir(data_path_1)[-2])
cell1_WBR431_20220316_121229_122234_0884k = os.path.join(data_path_1, os.listdir(data_path_1)[3])

test_examples = list(np.arange(5))

full_cell1_WBR431_20220316_004618_005623_0392k = []
for idx, csv in enumerate(os.listdir(cell1_WBR431_20220316_004618_005623_0392k)):
    if idx in test_examples:
        full_dir = os.path.join(cell1_WBR431_20220316_004618_005623_0392k, csv)
        full_cell1_WBR431_20220316_004618_005623_0392k.append(pd.read_csv(full_dir, header=None, index_col=None)[[2, 3]])

full_cell1_WBR431_20220316_121229_122234_0884k = []
for idx, csv in enumerate(os.listdir(cell1_WBR431_20220316_121229_122234_0884k)):
    if idx in test_examples:
        full_dir  = os.path.join(cell1_WBR431_20220316_121229_122234_0884k, csv)
        full_cell1_WBR431_20220316_121229_122234_0884k.append(pd.read_csv(full_dir, header=None, index_col=None)[[2, 3]])


# %% Data preprocessing
