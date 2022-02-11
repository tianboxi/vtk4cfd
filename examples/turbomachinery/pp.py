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
   'solvarnames': {'rho':'Density', 'P':'Pressure', 
                  'V':'Velocity', 'M':'Mach', 
                  'rhoet':'EnergyStagnationDensity'},
   'domname_key':['blk*'],
}

sol = Grid(solfile, 'CGNS', options=options, plotoptions={})

#plotvarnames = ['P','U']
#nmercontour, merc, mercf = len(plotvarnames), [], []
#for ii in range(nmercontour):
#   fig, thisax = sol.makePlot()
#   merc.append(thisax)
#   mercf.append(fig)
#
##sol.computeVar(['V/Vinf','Cpt','Cp'])
#zloc = -0.5 
#sol.makeSlice([0,0,zloc], [0,0,-1.0],slicename='midspan',view='-z')
#sol.makeSlice([0,0,zloc], [0,0,-1.0],slicename='midspan_quad',view='-z', triangulize=False)
#sline_seeds = np.linspace([-1.0, -0.3, zloc], [-1.0, 0.3, zloc], 100)
#sol.plotStreamlineMultiBlk('U', sline_seeds, surface='midspan_quad', x_range=[-1.0, 1.0], idir='forward', slname='midspan')
#
#sol.plotContour('p', surface='midspan', ax=merc[0])
##,sol.plotContour('Cp', surface='midspan', ax=merc[2], clevels=[-6.0, 6.0])
#
##for axes in merc:
#sol.plotSavedStreamline('midspan', merc[0])
#
#dpi = 300.0
##for ifig, axes in enumerate(merc):
#merc[0].set_xlim([-0.5, 1.5])
#merc[0].set_ylim([-0.3, 0.3])
#mercf[0].savefig('contours/'+plotvarnames[0]+'.png', dpi=dpi, bbox_inches='tight')

