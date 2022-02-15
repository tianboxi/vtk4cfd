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

solfile = './*.cgns'
options ={
   'solvarnames': {'rho':'Density', 'p':'Pressure', 
                  'V':'Velocity', 'M':'Mach', 
                  'rhoet':'EnergyStagnationDensity'},
   'domname_key':['blk*'],
   'ref_vals':{'p':101325.0, 'pt':101325.0},
}

sol = Grid(solfile, 'CGNS', options=options, plotoptions={})
sol.computeVar(['vx','Vmag','pt/ptref', 'p/pref'])
sol.makeSlice([0.0,0,0.0], [1.0,0.0,0.0],slicename='s1',view='-x')
varmave, mdot = sol.massInt('s1', ['p','vx'])
print(varmave, mdot)

nmercontour, merc, mercf = 1, [], []
for ii in range(nmercontour):
   fig, thisax = sol.makePlot()
   merc.append(thisax)
   mercf.append(fig)
#
##sol.computeVar(['V/Vinf','Cpt','Cp'])
sol.makeSlice([0,0,0], [0,-np.sin(0.0873),np.cos(0.0873)],slicename='mer',view='-z')
sol.makeSlice([0,0,0], [0,-np.sin(0.0873),np.cos(0.0873)],slicename='mer_quad',view='-z', triangulize=False)
sline_seeds = np.linspace([-0.15, 0.09848, 0.01736], [-0.15, 0.2462, 0.0434], 20)
sol.plotStreamlineMultiBlk('Velocity', sline_seeds, surface=None, x_range=[-0.15, 0.2], idir='forward', slname='mer')
#
sol.plotContour('pt/ptref', surface='mer', ax=merc[0])
##,sol.plotContour('Cp', surface='midspan', ax=merc[2], clevels=[-6.0, 6.0])
#
#for axes in merc:
sol.plotSavedStreamline('mer', merc[0])
#
dpi = 300.0

mercf[0].savefig('contours/'+'pt'+'.png', dpi=dpi, bbox_inches='tight')

