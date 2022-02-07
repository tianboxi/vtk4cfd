# =============================================================================
# Imports
# =============================================================================
#import vtk 
#import csv 
#import glob
#import json
#import math as m
#from math import pow 
#from math import sqrt
#import matplotlib.tri as tri 
#from mpl_toolkits.axes_grid1 import make_axes_locatable
#from scipy import interpolate
#from intersect import intersection
#from mpl_toolkits.mplot3d import Axes3D
#from collections import OrderedDict
#from .basepp import BCase
#from .airfoilpp import ACase
#from operator import itemgetter
#from matplotlib.lines import Line2D
import os
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import vtk4cfd as mv
from vtk4cfd import Grid

solfile = './tiny_1933.vtk'
options ={
  'solvarnames':{'P':'p', 'V':'U'}
}
sol_obj = Grid(solfile, 'VTK', options=options, plotoptions={})
