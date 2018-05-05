
import numpy as np
import sys, os
import matplotlib.pyplot as plt 

from qslam.slampf import ParticleFilter
from qslam.mapping import TrueMap

from contextlib import contextmanager

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

