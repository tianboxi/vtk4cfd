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
# Options
options ={
   'solvarnames': {'rho':'Density', 'p':'Pressure', 
                  'V':'Velocity', 'M':'Mach', 
                  'rhoet':'EnergyStagnationDensity'},
   'domname_key':['blk*'],
   'ref_vals':{'p':101325.0, 'pt':101325.0},
}
# create sol object
sol = Grid(solfile, 'CGNS', options=options, plotoptions={})
# compute variables
sol.computeVar(['vx','Vmag','pt/ptref', 'p/pref'])
# slice the volume solution
sol.makeSlice([0.0,0,0.0], [1.0,0.0,0.0],slicename='s1',view='-x')
# perform mass integration
varmave, mdot = sol.massInt('s1', ['p','vx'])
print(varmave, mdot)

# plot contours
nmercontour, merc, mercf = 1, [], []
for ii in range(nmercontour):
   fig, thisax = sol.makePlot()
   merc.append(thisax)
   mercf.append(fig)

#y
#^ ang
#|  v  /
#|    /
#|   /\
#|  /    \ 
#| /        \
#|/           n
# -------------> z
ang = 10.0/180.0*np.pi
sol.makeSlice([0,0,0], [0,-np.sin(ang),np.cos(ang)],slicename='mer',view='-z')
sol.makeSlice([0,0,0], [0,-np.sin(ang),np.cos(ang)],slicename='mer_quad',view='-z', triangulize=False)
sline_seeds = np.linspace([-0.15, 0.1*np.cos(ang), 0.1*np.sin(ang)], [-0.15, 0.25*np.cos(ang), 0.25*np.sin(ang)], 50)
sol.plotStreamlineMultiBlk('Velocity', sline_seeds, surface='mer_quad', x_range=[-0.15, 0.2], idir='forward', slname='mer', maxlength=0.2)
#
sol.plotContour('pt/ptref', surface='mer', ax=merc[0])
##,sol.plotContour('Cp', surface='midspan', ax=merc[2], clevels=[-6.0, 6.0])
#
#for axes in merc:
sol.plotSavedStreamline('mer', merc[0])
#
dpi = 300.0

mercf[0].savefig('contours/'+'pt'+'.png', dpi=dpi, bbox_inches='tight')

# plot lines
