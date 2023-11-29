import numpy as np
import os
from multiprocessing import get_context
import yaml
from scipy.interpolate import RegularGridInterpolator as rgi
import matplotlib.pyplot as plt
import argparse


x = np.linspace(0,5,20)
y = np.linspace(0,5,20)
z = np.linspace(0,5,20)
xlower = np.linspace(0,1,4)
ylower = np.linspace(0,1,4)
zlower = np.linspace(0,1,4)

data = 2 * np.random.random((4,4)) - 1

f = rgi((xlower, ylower), data, method='cubic')
