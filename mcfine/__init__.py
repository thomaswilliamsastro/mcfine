import multiprocessing as mp
import os

import matplotlib

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14
os.environ["OMP_NUM_THREADS"] = "1"
mp.set_start_method("fork")
