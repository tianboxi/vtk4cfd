import numpy as np
import vtk4cfd.myvtk as mv
import vtk4cfd.utils as utils
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.tri as tri
from mpl_toolkits.axes_grid1 import make_axes_locatable
from vtk.util.numpy_support import vtk_to_numpy
from scipy.interpolate import griddata
from collections import Counter
import copy
import math
import math as m
import os
import vtk
import glob


class Grid():

   def __init__(self, filename, filetype='VTK', datatype='CELL', 
                options=None, plotoptions=None):
      """
      input:
      ---
      filename: a list of vtk file or a single cgns file
      filetype: choice of "VTK" or "CGNS"
      datatype: choice of "CELL" or "POINT"
      options: dictionary of options
      plotoptions: dictionary of plot options
      """
      # set options
      self.fileName = filename
      self.caseOptions = self.setDefaultOptions(options)
      self.plotOptions = self.setDefaultPlotOptions(options)
      self.tstrs, self.lstrs = self.setDefaultStrings()
      # this is where main data file is linked to
      self.data = None
      self.drange = {}
      # variable existed in obj 
      self.vars = []
      # if this is an overset grid
      overset = self.caseOptions['overset']
      # import data
      if filetype == 'VTK':
         self.importVTK(filename, datatype, overset)
      elif filetype == 'CGNS':
         self.importCGNS(filename, datatype, overset)
      else:
         raise NotImplementedError
         print('Type of gridfile: '+filetype+'not implemented')

      # empty dics for data reusing 
      self.clips = {}
      self.cuts = {}
      self.slines = {}
      self.outlines = {}
      
      # reference state 
      self.refs = self.setRefState()
      # probe freestream properties
      if self.caseOptions['freestreamloc'] is not None:
         print('PROBE free stream properties at: ', self.caseOptions['freestreamloc'])
         self.infs = self.getFreeStreamProperties(datatype='POINT')     
      # range of data
      self.updateDRange()

      # perform grid transformation if requried 
      #if 'transform' in options:
      #   if options['transform'] is not None:
      #      self.transform(options['transform']['translate'], options['transform']['rotate'])

   ## ===== Initialization Functions ===== ##

   def setDefaultOptions(self, options):
      copt = {
      'overset': False,
      'refstate': None,
      'domlist': [],
      'domname_key': ['bw02_*.vtk'],
      'freestreamloc': None,
      'solvarnames': {'rho':'Density', 'p':'Pressure', 'T':'Temperature',
                            'V':'Velocity', 'M':'Mach',  
                            'rhoet':'EnergyStagnationDensity'},
      'ref_vals':{'p':101325.0}
      } 
      non_def_ops = []
      warning = False
      for op in options:
         if op in copt:
            copt[op] = options[op]
         else:
            non_def_ops.append(op)
            copt[op] = options[op]
            warning = True
      if warning:
         print('Some of the options has no default valuses in vtk4cfd')
      return copt

   def setDefaultPlotOptions(self, options):
      copt = {
              # contour plot options
              'cflevels':30,
              'clevels':30,
              'clcolors':'k',
              'cmap':'coolwarm',
              'cextend':'both',
              # color bar options
              'cbar_size':'3%',
              'cbar_loc':'right', 
              'cbar_pad':0.05,
              # streamline options
              'sline_spec':{'w':0.2, 'spec':'-k'},
              # general outline options
              'outline_spec':{'w':0.2, 'spec':'-k'},
             }
      for op in options:
         if op in copt:
            copt[op] = options[op]
         else:
            copt[op] = options[op]
      return copt

   def setRefState(self):
      refstate_input = self.caseOptions['refstate']
      if refstate_input is not None:
         pRef = refstate_input['pRef']
         TRef = refstate_input['TRef']
         rhoRef = pRef/(287.05*TRef)

         uRef = np.sqrt(pRef/rhoRef)

         print('REF states: P=', pRef, ', T=', TRef, 
               ', rho=', rhoRef, ', u=', uRef)

         refstate = {'pRef':pRef, 'TRef':TRef, 'rhoRef':rhoRef,
                     'uRef':uRef}

         # Re-Dimensionalize and re-name all vars
         varnames = self.caseOptions['solvarnames']
         rho = vtk_to_numpy(mv.GetArray(self.data,varnames['rho']))*rhoRef
         v = vtk_to_numpy(mv.GetArray(self.data,varnames['V']))*uRef
         rhoet = vtk_to_numpy(mv.GetArray(self.data,varnames['rhoet']))*rhoRef*uRef**2
         P = vtk_to_numpy(mv.GetArray(self.data,varnames['p']))*pRef
         M = vtk_to_numpy(mv.GetArray(self.data,varnames['M']))
         self.data = mv.RemovePArray(self.data, varnames['rho'])
         self.data = mv.RemovePArray(self.data, varnames['p'])
         self.data = mv.RemovePArray(self.data, varnames['V'])
         self.data = mv.RemovePArray(self.data, varnames['M'])
         self.data = mv.RemovePArray(self.data, varnames['rhoet'])
         self.data = mv.AddNewArray(self.data, rho, varnames['rho'])
         self.data = mv.AddNewArray(self.data, P, varnames['p'])
         self.data = mv.AddNewArray(self.data, v, varnames['V'])
         self.data = mv.AddNewArray(self.data, M, varnames['M'])
         self.data = mv.AddNewArray(self.data, rhoet, varnames['rhoet'])
         self.vars = self.vars + [varnames['rho'], varnames['p'], varnames['V'], varnames['M'], varnames['rhoet']]
         

         
      else:
         refstate = None


      return refstate

   def updateDRange(self):
      """
      Get range of all existing variables
      """
      for var in self.vars:
         array = mv.GetArray(self.data, var)
         array_range = array.GetFiniteRange(-1)
         self.drange[var] = array_range
         #print('Data Range: ', self.drange)
         
   def getFreeStreamProperties(self, datatype='POINT',
                               propertyNames=None):
      """
      Probe a spcific point for freestream properties
      --
      propertyNamse:[pressure_name, velocity name, density name, mach number name]
      """
      loc = self.caseOptions['freestreamloc']
      pN = propertyNames
      if pN is None:
         varnames = self.caseOptions['solvarnames']
         if 'rho' in varnames: 
            pN = [varnames['p'], varnames['V'], varnames['rho'], varnames['M']]
         else: 
            pN = [varnames['p'], varnames['V']]

      if datatype=='POINT':
         src = self.data
      else:
         src = self.datac

      comp = True
      if len(pN) < 4: # incompressible flow 'p' 'V' only
         comp = False
         
      infState = self.probFlowField([loc], pN, source=src)
      infState[pN[1]] =  [np.linalg.norm(infState[pN[1]])]
      for state in infState:
         infState[state] = infState[state][0]
      if comp:
         infState['pt'] =  infState[pN[0]]/((1+(1.4-1)/2*infState[pN[3]]**2)**(-1.4/(1.4-1)))
         infState['T'] = infState[pN[0]]/(287.0*infState[pN[2]])

      print('INF states: ', infState)
      return infState

   ## ===== Plotting functions ========= ##
   
   def plotContour(self, varname, surface, ax=None, 
                   cbar=True, clabel=None, cbar_wid='3%', 
                   axlim=None, cline=None, clevels=None):

      cut = self.cuts[surface]['vtkcut']
      triangulation = self.cuts[surface]['triangulation']
      normal = self.cuts[surface]['normal']
      defvarname = self.caseOptions['solvarnames']
      if varname in defvarname:
         varname = defvarname[varname]

      vartoplot = vtk_to_numpy(mv.GetArray(cut, varname))

      if not ax:
         fig, ax = self.makePlot(axlim=axlim)
      else:
         fig = ax.get_figure()

      
      if cline is not None:
         nline = cline['nline']
         color = cline['lcolor']
         drange = self.drange[varname]
         levels = np.linspace(drange[0], drange[1], nline)
         cs = ax.tricontour(triangulation, vartoplot, levels, colors='k',
         linewidths=0.1, linestyles='solid')
         #ax.clabel(cs, fontsize=6, inline=1)

      if clevels is not None:
         vmin = clevels[0]
         vmax = clevels[1]
         levels = np.linspace(vmin,vmax,30)
         print(varname+':'+'SET min='+str(vmin)+', max='+str(vmax))
      else:
         levels = self.plotOptions['cflevels']

      if clabel is None:
         clabel = varname

      csf = ax.tricontourf(triangulation, vartoplot, levels=levels, 
            cmap=self.plotOptions['cmap'], extend=self.plotOptions['cextend'])

      if cbar:
         # create an axes on the right side of ax. The width of cax will be 3%
         # of ax and the padding between cax and ax will be fixed at 0.05 inch.
         divider = make_axes_locatable(ax)
         if self.plotOptions['cbar_loc'] == 'right':
            cax = divider.append_axes("right", size=cbar_wid, 
            pad=self.plotOptions['cbar_pad'])
            # set color bar, extendrect=true to avoid pointy ends
            cbar = fig.colorbar(csf, cax=cax, orientation='vertical', 
                   label=clabel,extendrect=True)
         elif self.plotOptions['cbar_loc'] == 'bottom':
            cax = divider.append_axes("bottom", size=cbar_wid, 
            pad=self.plotOptions['cbar_pad'])
            cbar = fig.colorbar(csf, cax=cax, orientation='horizontal', 
                   label=clabel,extendrect=True)

      ax.set_aspect('equal')

   def plotVectors_old(self, varname, surface, ax, view, flip):
      """
      Plot vectors 
      """
      cut = self.cuts[surface]['vtkcut']
      #view = self.cuts[surface]['view']
      nodes = vtk_to_numpy(mv.GetNodes(cut))
      vartoplot = vtk_to_numpy(mv.GetArray(cut, varname))
      #TODO mask some of the points
      #for ii in range(nodes.shape[0]):
      x,y,z = nodes[:,0], nodes[:,1], nodes[:,2]
      vx, vy, vz = vartoplot[:,0], vartoplot[:,1], vartoplot[:,2]
      if view == '-x':
         x, y = nodes[:,1], nodes[:,2]
         vx, vy = vartoplot[:,1], vartoplot[:,2]
      elif view == '+x':
         x, y = -1*nodes[:,1], nodes[:,2]
         vx, vy = -1*vartoplot[:,1], vartoplot[:,2]
      elif view == '-y':
         x, y = -1*nodes[:,0], nodes[:,2]
         vx, vy = -1*vartoplot[:,0], vartoplot[:,2]
      elif view == '+y':
         x, y = nodes[:,0], nodes[:,2]
         vx, vy = vartoplot[:,0], vartoplot[:,2]
      elif view == '-z':
         x, y = nodes[:,0], nodes[:,1]
         vx, vy = vartoplot[:,0], vartoplot[:,1]
      elif view == '+z':
         x, y = -1*nodes[:,0], nodes[:,1]
         vx, vy = -1*vartoplot[:,0], vartoplot[:,1]
      if flip:
         xplot, yplot = y, x
         vxplot, vyplot = vy, vx
      else:
         xplot, yplot = x, y
         vxplot, vyplot = vx, vy
      # the smaller the scale the longer the arrow
      ax.quiver(xplot[::100],yplot[::100],vxplot[::100],vyplot[::100],scale=300.0)

   def plotUniformVectors(self, varname, allpts, view, flip, ax, 
             width = 0.002, scale=300.0, normalized=False):

      #x_range, yrange = xyrange[0] , xyrange[1]
      #x = np.linspace(x_range[0], x_range[1], nx)
      #y = np.linspace(y_range[0], y_range[1], ny)
      #for xx in x:
      #   for yy in y:
      #      pt = 
      defvarname = self.caseOptions['solvarnames']
      if varname in defvarname:
         varname = defvarname[varname]
      pdata = self.probFlowField(allpts, [varname])
      vals = np.asarray(pdata[varname])
      px, py = self.getPlotXY(np.asarray(allpts), view, flip)
      vx, vy = self.getPlotXY(np.asarray(vals), view, flip)

      ax.quiver(px, py, vx, vy, scale=300.0, width=width)
      

   def plotVectors(self, varname, surface, ax, view, flip, shaftwidth = 1.0/1500.0,
                   scale=300.0, everyN=100, xyrange=None, normalized=False):
      """ 
      Plot vectors 
      """
      cut = self.cuts[surface]['vtkcut']
      #view = self.cuts[surface]['view']
      nodes = vtk_to_numpy(mv.GetNodes(cut))
      defvarname = self.caseOptions['solvarnames']
      if varname in defvarname:
         varname = defvarname[varname]
      vartoplot = vtk_to_numpy(mv.GetArray(cut, varname))
      #TODO mask some of the points
      #for ii in range(nodes.shape[0]):
      x,y,z = nodes[:,0], nodes[:,1], nodes[:,2]
      vx, vy, vz = vartoplot[:,0], vartoplot[:,1], vartoplot[:,2]
      if view == '-x' or view ==  'front':
         x, y = -1*nodes[:,2], nodes[:,1]
         vx, vy = -1*vartoplot[:,2], vartoplot[:,1]
      elif view == '+x' or view == 'back':
         x, y = nodes[:,2], nodes[:,1]
         vx, vy = vartoplot[:,2], vartoplot[:,1]
      elif view == '-y' or view == 'top':
         x, y = nodes[:,0], -1*nodes[:,2]
         vx, vy = vartoplot[:,0],-1*vartoplot[:,2]
      elif view == '+y' or view == 'bottom':
         x, y = nodes[:,0], nodes[:,2]
         vx, vy = vartoplot[:,0], vartoplot[:,2]
      elif view == '-z' or view == 'right':
         x, y = nodes[:,0], nodes[:,1]
         vx, vy = vartoplot[:,0], vartoplot[:,1]
      elif view == '+z' or view == 'left':
         x, y = nodes[:,0], -1*nodes[:,1]
         vx, vy = vartoplot[:,0], -1*vartoplot[:,1]
      if flip:
         xplot, yplot = y, x
         vxplot, vyplot = vy, vx
      else:
         xplot, yplot = x, y
         vxplot, vyplot = vx, vy
      # do normalization
      if normalized:
         ll = np.sqrt(np.asarray(vxplot)**2 + np.asarray(vyplot)**2)
         vxplot = vxplot/ll
         vyplot = vyplot/ll
      # apply mask
      if xyrange is not None:
         # create a masking array for each point
         mask = np.zeros(len(nodes), dtype=bool)
         for ip, point in enumerate(zip(xplot, yplot)):
            mask[ip] = True
            if point[0] > xyrange[0][0] and point[0] < xyrange[0][1]:
               if point[1] > xyrange[1][0] and point[1] < xyrange[1][1]: 
                  mask[ip] = False 
         xplot = np.ma.compressed(np.ma.masked_array(xplot, mask=mask))
         yplot = np.ma.compressed(np.ma.masked_array(yplot, mask=mask))
         vxplot = np.ma.compressed(np.ma.masked_array(vxplot, mask=mask))
         vyplot = np.ma.compressed(np.ma.masked_array(vyplot, mask=mask))
            
      # the smaller the scale the longer the arrow
      # linewidth controls quiver thickness
      nn = everyN
      #ax.quiver(xplot[::nn],yplot[::nn],vxplot[::nn],vyplot[::nn], scale=scale, width=shaftwidth)
      ax.quiver(xplot[::nn],yplot[::nn],vxplot[::nn],vyplot[::nn], scale=scale)


   def plotStreamlineMultiBlk(self, varname, points, x_range, 
                           surface=None, ax=None, 
                           idir='both', slname=None, view=None, maxlength=2.0, maxstep=None):
      """
      Plot streamline for multi-region or overset grids  
      (When discontinuities might exist in between blks)
      """
      if surface:
         source = self.cuts[surface]['vtkcut']
         view = self.cuts[surface]['view']
         planar = True
      else:
         source = self.data
         view = view
         planar = False
         
      defvarname = self.caseOptions['solvarnames']
      if varname in defvarname:
         varname = defvarname[varname]

      #nodes = vtk_to_numpy(mv.GetNodes(source))
      npt = len(points)

      allslines = []
      for i, thisPt in enumerate(points):
         sline = mv.Streamline(source, varname, thisPt, idir=idir, planar=planar)
         if len(sline)>0:
            endx = sline[-1,0]
            counter = 0
            if idir == 'forward':
               while endx < x_range[1]:
                  pt_next = sline[-1,:] + (sline[-1,:] - sline[-2,:])*2.0
                  sline_next = mv.Streamline(source, varname, pt_next, idir='forward', planar=planar, maxlength=maxlength, maxstep=maxstep)
                  sline = np.concatenate((sline, sline_next), axis=0)
                  endx = sline[-1,0]
                  counter = counter + 1
                  if len(sline_next)==1:
                     print('streamline cannot continue')
                     break
                  if counter > 500:
                     print('too many iterations')
                     break
            elif idir == 'backward':
               while endx > x_range[0]:
                  pt_next = sline[-1,:] - (sline[-1,:] - sline[-2,:])*2.0
                  sline_next = mv.Streamline(self.data, varname, pt_next, idir='backward', planar=planar, maxlength=maxlength, maxstep=maxstep)
                  sline = np.concatenate((sline, sline_next), axis=0)
                  endx = sline[-1,0]
                  counter = counter + 1
                  if len(sline_next)==1:
                     print('streamline cannot continue')
                     break
                  if counter > 500:
                     print('too many iterations')
                     break
            if ax is not None: 
               ax.plot(sline[:-1,0],sline[:-1,1], 
                       self.plotOptions['sline_spec']['spec'],
                       linewidth=self.plotOptions['sline_spec']['w'])
               ax.set_aspect('equal')

            allslines.append(sline)

         else:
            print('No streamline found at ',thisPt)

      if slname:
         self.slines[slname] = allslines

      return allslines


   def getPlotXY(self, nodes, view, flip):
      # look against the stream
      if view == '-x' or view ==  'front':
         x, y = -1*nodes[:,2], nodes[:,1]
      # look along the stream
      elif view == '+x' or view == 'back':
         x, y = nodes[:,2], nodes[:,1]
      # look from top
      elif view == '-y' or view == 'top':
         x, y = nodes[:,0], -1*nodes[:,2]
      # look from bottom
      elif view == '+y' or view == 'bottom':
         x, y = nodes[:,0], nodes[:,2]
      # look from right (to the stream)
      elif view == '-z' or view == 'right':
         x, y = nodes[:,0], nodes[:,1]
      # look from left 
      elif view == '+z' or view == 'left':
         x, y = nodes[:,0], -1*nodes[:,1]
      if flip:
         xplot, yplot = y, x
      else:
         xplot, yplot = x, y
      return xplot, yplot

   def makeSlice(self, center, normal, view='-z', flip=False, slicename=None, source=None, 
                 saveboundary=False, triangulize=True):
      """
      make a slice cut of the flow domain  
      """
      if source is None:
         source = self.data
      plane = mv.CreatePlaneFunction(center, normal)
      cut = mv.Cut(source,plane,tri=triangulize)
      # Get triangles from vtk cut 
      triangles = np.asarray(mv.GetTriangles(cut))
      # Reproduce triangulation with matplotlib
      nodes = vtk_to_numpy(mv.GetNodes(cut))
      ## TODO rotate to a view such that the 2D plane is x-y plane (looking against z)
      ## toe, pitch angles
      #pitch = np.arctan2(axis[2], axis[0]) - np.pi/2.0
      #toe = np.arctan2(axis[1], axis[0])
      #roll = np.arctan2(axis[2], axis[1]) - np.pi/2.0
      ## do the transformation
      # Set viewing direction
      xplot, yplot = self.getPlotXY(nodes, view, flip)
      # Convert 3D nodes to 2D coord system on the cut surface
      triangulation = tri.Triangulation(xplot, yplot, triangles)
      # save cut for later use if slicename is specified
      if slicename:
         self.cuts[slicename] = \
         {'vtkcut':cut, 'triangulation':triangulation, 'view':view, 'normal':normal}

      return cut, triangulation

   #def makeClip(self, clip_limit, center, normal, clipname=None, source=None):
   #   """
   #   make a clip
   #   ---
   #   input param:
   #   
   #   clip_limit: list, [xmin, xmax, ymin, ymax]
   #   """
   #   clip_plane = []
   #   clip_plane.append(mv.CreatePlaneFunction((clip_limit[0],0.0,0.5),(1,0,0)))
   #   clip_plane.append(mv.CreatePlaneFunction((clip_limit[1],0.0,0.5),(-1,0,0)))
   #   clip_plane.append(mv.CreatePlaneFunction((0.0,clip_limit[2],0.5),(0,1,0)))
   #   clip_plane.append(mv.CreatePlaneFunction((0.0,clip_limit[3],0.5),(0,-1,0)))
   #   data = self.data
   #   for clip in clip_plane:
   #      data = mv.Clip(data, clip)

   #   self.clips[clipname] = data

   def makeClip(self, center, normal, clipname=None, source=None, data=None):
      """
      make a clip
      ---
      input param:
      """
      if data is None:
         data = self.data

      clip_plane = mv.CreatePlaneFunction(center,normal)
      data_clip = mv.Clip(data, clip_plane)
      if clipname is not None:
         self.clips[clipname] = data_clip
      
      return data_clip

   def makeContourSurfs(self, varnames, drange):
      """
      make contour surface of a certain variable that exists in the solution 
      """
      pass

   ## ====== Geometry handeling ====== ##
   #def plotAlphaShapes(self, surface, triangulation, ax=None):
   #   """
   #   An implementation of alpha shape through a surface with existing triangulations
   #   (Could also be done via Delaunay Triangulation)
   #   See: https://stackoverflow.com/questions/23073170/calculate-bounding-polygon-of-alpha-shape-from-the-delaunay-triangulation

   #   Input: 
   #   ---
   #   surface, a vtk object (same as that of a cut plane) 
   #   ax, matplotlib object
   #   """

   #   nodes = vtk_to_numpy(mv.GetNodes(surface))
   #   conn, neighbor = mv.GetTriangles(surface, neighbor=True)
   #   edges = triangulation.edges
   #   print(edges)
   #   unique_edges = findBoundaryEdges(edges)
   #   print(len(unique_edges))
   #   for edge in unique_edges:
   #      ind1, ind2 = edge[0], edge[1]
   #      pts = np.asarray([[nodes[ind1,0], nodes[ind1,1]], 
   #                        [nodes[ind2,0], nodes[ind2,1]]])
   #      ax.plot(pts[:,0], pts[:,1], '-k')


   def getMeshBoundaries(self, source=None, surface=None, ax=None, name=None):
      """
      get the boundary curves of a mesh (vtkFeatureEdges)
      """
      if source is None:
         source = self.data
      #if surface:
      #   data = self.cuts[surface]['vtkcut']
      #   edge_pts = mv.GetFeatureEdges(data)
      #if mesh:
      #   data = self.clips[mesh]

      edge_pts = mv.GetFeatureEdges(source, surface=surface)
      if name is not None:
         self.outlines[name] = edge_pts
      if ax is not None:
         for edge in edge_pts:
            x = [edge[0][0], edge[1][0]]
            y = [edge[0][1], edge[1][1]]
            ax.plot(x, y, '-k')

   def plotSavedOutlines(self, name, ax, view='right', flip=False): 
      for edge in self.outlines[name]:
         #x = [edge[0][0], edge[1][0]]
         #y = [edge[0][1], edge[1][1]]
         this_edge = np.asarray([edge[0],edge[1]])
         x, y = self.getPlotXY(this_edge, view, flip)
         ax.plot(x, y, self.plotOptions['outline_spec']['spec'],
                 linewidth=self.plotOptions['outline_spec']['w'])

   def plotSavedStreamline(self, name, ax, view='right', flip=False): 
      slines = self.slines[name]
      for sline in slines:
         this_line = np.asarray(sline)
         x, y = self.getPlotXY(this_line, view, flip)
         #ax.plot(sline[:-1,0],sline[:-1,1], 
         ax.plot(x, y, self.plotOptions['sline_spec']['spec'],
         linewidth=self.plotOptions['sline_spec']['w'])

   def plotGridLines(self,triangulation,ax):
      ax.triplot(triangulation)


   ## ====== CFD data processing ======= ##
   def transform(self, translate, rotate):
      """
      transform dataset
      """
      data = self.data
      data_trans = mv.Transform(data, translate, rotate)
      self.data = data_trans

      return data_trans

   def probFlowField(self, points, varlist, source=None):
      """
      prob flow field variables at a list of locations 
      """
      if source is None:
         source = self.data
      tmp = []
      for pt in points:
         vtkpt = mv.CreatePointSource(pt, radius=0.0, numberofpoints=1)
         probe = mv.Probe(source, vtkpt)
         probevar = []
         for var in varlist:
            probevar.append(vtk_to_numpy(mv.GetArray(probe,var))[0])
         tmp.append(probevar)

      probeResult = {}
      for ivar, var in enumerate(varlist):
         probeResult[var] = []
         for ii in range(len(points)):
            probeResult[var].append(tmp[ii][ivar])

      return probeResult

   def getVar(self, varname, datatype='POINT'):
      """
      get raw data of a flow variable 
      """
      if datatype == 'POINT':
         source = self.data
         data = vtk_to_numpy(mv.GetArray(source, varname))
      elif datatype == 'CELL':
         source = self.datac
         data = vtk_to_numpy(mv.GetCArray(source, varname))

      return data

   def computeCustomVar(self, inputvars, outputvar, compute_fun):
      """
      compute a customized variable by passing a function: compute_fun
      """
      data = self.data
      allvars = self.vars
      inputvar_dic = {}
      def_varnames = self.caseOptions['solvarnames']
      for varname in inputvars:
         if varname in def_varnames:
            inputvar_dic[varname] = vtk_to_numpy(mv.GetArray(data, def_varnames[varname]))
         elif varname in allvars: 
            inputvar_dic[varname] = vtk_to_numpy(mv.GetArray(data, varname))
         else:
            print(varname+' DO NOT EXIST!!!!')

      outputvar_array = compute_fun(inputvar_dic, self)

      if outputvar not in self.vars:
         data = mv.AddNewArray(data, outputvar_array, outputvar)      
         self.vars.append(outputvar)


   def computeVar(self, varlist):
      """
      Given the basic flow solutions, compute other flow variables 
      """
      # import known states as numpy array
      data = self.data
      nodes = vtk_to_numpy(mv.GetNodes(data))
      varnames = self.caseOptions['solvarnames']
      if 'rho' in varnames:
         rho = vtk_to_numpy(mv.GetArray(data,varnames['rho']))
      if 'V' in varnames:
         v = vtk_to_numpy(mv.GetArray(data,varnames['V']))
      if 'rhoet' in varnames:
         rhoet = vtk_to_numpy(mv.GetArray(data,varnames['rhoet']))
      if 'p' in varnames:
         P = vtk_to_numpy(mv.GetArray(data,varnames['p']))
      if 'M' in varnames:
         mach = vtk_to_numpy(mv.GetArray(data,varnames['M']))
      #T = vtk_to_numpy(mv.GetArray(data,'Temperature'))*self.TRef
      if 'p' in varnames and 'rho' in varnames:
         T = P/(rho*287.0)
      if 'rhoet' in varnames:
         Et = rhoet/rho
      if 'rhoet' in varnames:
         Ht = Et+P/rho
      if 'rhoet' in varnames: 
         Pt = P/((1+(1.4-1)/2*mach**2)**(-1.4/(1.4-1)))    # total pressure from isentropic relation

      N = len(rho)

      nodes_cyl = utils.CartToCyl(nodes,'coord')        # convert point coords to cylindrical system
      v_cyl = utils.CartToCyl(v,'vector', nodes_cyl[:,1])  # Velocity in cyl

      if 'RPM' in self.caseOptions:
         omega = self.caseOptions['RPM']*2*np.pi/60.0
         w_cyl = np.zeros((N,3))                        # Compute relative velocity
         w_cyl[:,0] = v_cyl[:,0]
         w_cyl[:,1] = v_cyl[:,1] + omega*nodes_cyl[:,0]
         w_cyl[:,2] = v_cyl[:,2]
         w = utils.CylToCart(w_cyl, 'vector', nodes_cyl[:,1])
         if 'rhoet' in varnames:
            machr = np.sqrt(w[:,0]**2+w[:,1]**2+w[:,2]**2)/np.sqrt(1.4*287*T)

      # cylindrical system coords (for verification)
      if 'rr' in varlist and 'rr' not in self.vars:
         data = mv.AddNewArray(data,nodes_cyl[:,0],'rr')      
         self.vars.append('rr')
      if 'theta' in varlist and 'theta' not in self.vars:
         data = mv.AddNewArray(data,nodes_cyl[:,1],'theta')      
         self.vars.append('theta')
      # Basic flow variables 
      if 'vx' in varlist and 'vx' not in self.vars:
         data = mv.AddNewArray(data,v[:,0],'vx')      
         self.vars.append('vx')
      if 'vy' in varlist and 'vy' not in self.vars:
         data = mv.AddNewArray(data,v[:,1],'vy')
         self.vars.append('vy')
      if 'vz' in varlist and 'vz' not in self.vars:
         data = mv.AddNewArray(data,v[:,2],'vz')
         self.vars.append('vz')
      if 'vt' in varlist and 'vt' not in self.vars:
         data = mv.AddNewArray(data,v_cyl[:,1],'vt')
         self.vars.append('vt')
      if 'vr' in varlist and 'vr' not in self.vars:
         data = mv.AddNewArray(data,v_cyl[:,0],'vr')
         self.vars.append('vr')
      if 'et' in varlist and 'et' not in self.vars:
         et = rhoet/rho
         data = mv.AddNewArray(data,et,'et')
         self.vars.append('et')
      if 'ht' in varlist and 'ht' not in self.vars:
         Ht = rhoet/rho+P/rho
         data = mv.AddNewArray(data,Ht,'ht')
         self.vars.append('ht')
      if 'T' in varlist and 'T' not in self.vars:
         data = mv.AddNewArray(data,T,'T')      
         self.vars.append('T')
      if 'pt' in varlist and 'pt' not in self.vars:
         data = mv.AddNewArray(data,Pt,'pt')      
         self.vars.append('pt')
      if 'Tt' in varlist and 'Tt' not in self.vars:
         Tt = T/((1+(1.4-1)/2*mach**2)**(-1))              # total temperature
         data = mv.AddNewArray(data,Tt,'Tt')      
         self.vars.append('Tt')
      if 's' in varlist and 's' not in self.vars:
         #S = 287*(1/(1.4-1)*np.log(T)+np.log(1/rho))        # Entropy
         Tinf = self.infs['T']
         rhoinf = self.infs[varnames['rho']]
         S = 287*(1/(1.4-1)*np.log(T/Tinf)+np.log(rhoinf/rho))        # Entropy
         data = mv.AddNewArray(data,S,'s')      
         self.vars.append('s')
      if 'V/Vinf' in varlist and 'V/Vinf' not in self.vars:
         vmag = np.linalg.norm(v, axis=1)
         vovervinf = vmag/self.infs[varnames['V']]
         data = mv.AddNewArray(data,vovervinf,'V/Vinf')      
         self.vars.append('V/Vinf')
      if 'Cp' in varlist and 'Cp' not in self.vars:
         cp = (P-self.infs[varnames['p']])/(0.5*self.infs[varnames['rho']]*\
              self.infs[varnames['V']]**2)
         data = mv.AddNewArray(data,cp,'Cp')      
         self.vars.append('Cp')
      if 'Cpt' in varlist and 'Cpt' not in self.vars :
         cpt = (Pt-self.infs['pt'])/(0.5*self.infs[varnames['rho']]*\
               self.infs[varnames['V']]**2)
         data = mv.AddNewArray(data,cpt,'Cpt')      
         self.vars.append('Cpt')
      if 'Vmag' in varlist and 'Vmag' not in self.vars:
         vmag = np.linalg.norm(v, axis=1)
         data = mv.AddNewArray(data,vmag,'Vmag')      
         self.vars.append('Vmag')

      if 'pt/ptref' in varlist and 'pt/ptref' not in self.vars:
         ptref = self.caseOptions['ref_vals']['pt']
         data = mv.AddNewArray(data, Pt/ptref,'pt/ptref')      
         self.vars.append('pt/ptref')

      if 'p/pref' in varlist and 'p/pref' not in self.vars:
         pref = self.caseOptions['ref_vals']['p']
         data = mv.AddNewArray(data, P/pref,'p/pref')      
         self.vars.append('p/pref')

      if 'v/vref' in varlist and 'v/vref' not in self.vars:
         vref = self.caseOptions['ref_vals']['V']
         vmag = np.linalg.norm(v, axis=1)
         data = mv.AddNewArray(data,vmag/vref,'v/vref')      
         self.vars.append('v/vref')

      # == Turbomachinery Variables ==
      if 'alpha' in varlist and 'alpha' not in self.vars:
         alpha = np.arctan2(v_cyl[:,1],np.sqrt(v_cyl[:,2]**2+v_cyl[:,0]**2))/np.pi*180.0
         #*(v_cyl[:,2]/np.absolute(v_cyl[:,2])) )/np.pi*180.0
         #alpha = -np.arctan2(v_cyl[:,1],v_cyl[:,2])/m.pi*180
         data = mv.AddNewArray(data,alpha,'alpha')      
         self.vars.append('alpha')

      if 'beta' in varlist and 'beta' not in self.vars:
         beta = np.arctan2(w_cyl[:,1],np.sqrt(w_cyl[:,2]**2 + w_cyl[:,0]**2))/np.pi*180
         #beta = np.arctan2(w_cyl[:,1],w_cyl[:,2])/m.pi*180
         data = mv.AddNewArray(data,beta,'beta')      
         self.vars.append('beta')

      if 'vr' in varlist and 'vr' not in self.vars:
         data = mv.AddNewArray(data,v_cyl[:,0],'vr')      
         self.vars.append('vr')
      if 'vt' in varlist and 'vt' not in self.vars:
         data = mv.AddNewArray(data,v_cyl[:,1],'vt')      
         self.vars.append('vt')
      if 'W' in varlist and 'W' not in self.vars:
         data = mv.AddNewArray(data,w,'W')      
         self.vars.append('W')
      if 'Wmag' in varlist and 'Wmag' not in self.vars:
         wmag = np.linalg.norm(w, axis=1)
         data = mv.AddNewArray(data,wmag,'Wmag')      
         self.vars.append('Wmag')
      if 'Wt' in varlist and 'Wt' not in self.vars:
         data = mv.AddNewArray(data,w_cyl[:,1],'Wt')      
         self.vars.append('Wt')
      if 'Pmech' in varlist and 'Pmech' not in self.vars:
         Pmechloc = v[:,0]*(Pt-101325.0) # flow mechanical power local
         data = mv.AddNewArray(data,Pmechloc,'Pmech')      
         self.vars.append('Pmech')
      if 'PK' in varlist and 'PK' not in self.vars:
         vmag = np.linalg.norm(v, axis=1)
         PK = v[:,0]*((P+0.5*rho*vmag**2) - (self.infs[varnames['p']]+0.5*self.infs[varnames['rho']]*self.infs[varnames['V']]**2))
         data = mv.AddNewArray(data,PK,'PK')      
         self.vars.append('PK')
      if 'phi' in varlist and 'phi' not in self.vars:
         phi = v[:,0]/(self.caseOptions['tip_radius']*omega)
         data = mv.AddNewArray(data,phi,'phi')      
         self.vars.append('phi')

      self.updateDRange()

      return data

   def convertToRotationalFrame(self, center, axis, rpm):
      """
      Convert flow variables from stationary reference to a rotational frame
      """
      pass

   
   ## ====== Integrations ====== ##
   def massInt(self, surface, varnames):
      """
      Mass integration and mass averaging of other properties over a specified surface 
      """
      surf = self.cuts[surface]['vtkcut'] 
      triangulation = self.cuts[surface]['triangulation']
      normal = self.cuts[surface]['normal']
      defvarnames = self.caseOptions['solvarnames']

      rho = vtk_to_numpy(mv.GetArray(surf, defvarnames['rho']))
      V = vtk_to_numpy(mv.GetArray(surf, defvarnames['V']))
      # Velocity normal to the integration surface
      Vn = V[:,0]*normal[0] + V[:,1]*normal[1] + V[:,2]*normal[2]
      N = len(rho)

      intvar = np.zeros((N, len(varnames)))
      for i, name in enumerate(varnames):
         try:
            intvar[:,i] = vtk_to_numpy(mv.GetArray(surf, defvarnames[name]))
         except:
            intvar[:,i] = vtk_to_numpy(mv.GetArray(surf, name))

      ncell = surf.GetOutput().GetNumberOfCells()
      #print(surf.GetOutput().GetCell(100).TriangleCenter())
      masssum, varsum = 0.0, np.zeros(len(varnames))
      for j in range(ncell):
         area =surf.GetOutput().GetCell(j).ComputeArea()
         ids = []
         for i in range(3):
            ids.append(int(surf.GetOutput().GetCell(j).GetPointId(i)))
         averho = (rho[ids[0]]+rho[ids[1]]+rho[ids[2]])/3.0
         aveu = (Vn[ids[0]]+Vn[ids[1]]+Vn[ids[2]])/3.0
         masscell = averho*aveu*area
         avevar = []
         for k, name in enumerate(varnames):
            avevar.append((intvar[ids[0],k]+intvar[ids[1],k]+intvar[ids[2],k])/3.0)

         masssum = masssum + masscell
         varsum = varsum + masscell * np.array(avevar)

      varmassave = varsum/masssum

      return varmassave, masssum 
      
   
   def surfaceInt(self, surface, varnames):
      """
      Surface integration of flow properties
      """
      surf = self.cuts[surface]['vtkcut'] 
      triangulation = self.cuts[surface]['triangulation']
      normal = self.cuts[surface]['normal']
      defvarnames = self.caseOptions['solvarnames']

      nodes = vtk_to_numpy(mv.GetNodes(surf))
      N = len(nodes)

      intvar = np.zeros((N, len(varnames)))

      for i, name in enumerate(varnames):
         try:
            intvar[:,i] = vtk_to_numpy(mv.GetArray(surf, defvarnames[name]))
         except:
            intvar[:,i] = vtk_to_numpy(mv.GetArray(surf, name))

      ncell = surf.GetOutput().GetNumberOfCells()
      varsum = np.zeros(len(varnames))
      for j in range(ncell):
         area =surf.GetOutput().GetCell(j).ComputeArea()
         ids = []
         for i in range(3):
            ids.append(int(surf.GetOutput().GetCell(j).GetPointId(i)))
         avevar = []
         for k, name in enumerate(varnames):
            avevar.append((intvar[ids[0],k]+intvar[ids[1],k]+intvar[ids[2],k])/3.0)

         varsum = varsum + area * np.array(avevar)

      return varsum 

   def volInt(self, varname):
      """
      Volume integration
      """
      pass


   ## ====== Import export functions ====== ##
   def importCGNS(self, filename, datatype, overset):
      """
      Import cgns files 
      Note you must have cgnsTools installed which includes 'cgns_to_vtk'
      """
      mycmd = 'cgns_to_vtk -a '+ filename
      try:
         os.system(mycmd)
      except:
         print('Error converting cgns to vtk, check if you have cgnsTool installed')
         return False
      domlist = self.caseOptions['domlist']
      if len(domlist) == 0:
         print('Domain names not specified in the case options thruough: domlist, try reading .vtk files in current dir')
         all_domname_key = self.caseOptions['domname_key'] 
         for domname_key in all_domname_key:
            print('Found domains: ', glob.glob(domname_key))
            for vtkfile in glob.glob(domname_key):
               domlist.append(vtkfile) 
         #return False

      self.caseOptions['nblk']= len(domlist)
      self.data_doms = []
      # loop through cgns doms
      for dom in domlist:
         self.data_doms.append(mv.ReadVTKFile(dom))
         # remove converted VTK file
         mycmd = 'rm '+dom
         os.system(mycmd)

      if len(domlist)>1:
         self.data = mv.combineDom(self.data_doms)
         # Note combined dom will have to be unstructured grid
      else:
         self.data = self.data_doms[0]

      # Determine type of grid

      if overset:
         # Only keep compute cells
         self.data = mv.threshold(self.data, 'Iblank', [1.0,1.0])

      # convert all cell data to point data
      if datatype is 'CELL':
         self.datac = vtk.vtkUnstructuredGrid()
         self.datac.DeepCopy(self.data.GetOutput())
         self.data = mv.CtoP(self.data)

      self.ncell=  self.data.GetOutput().GetNumberOfCells()
      self.npoints=self.data.GetOutput().GetNumberOfPoints()
      self.narray =self.data.GetOutput().GetPointData().GetNumberOfArrays()
      self.arrayNames = []

      for ii in range(self.narray):
         name = self.data.GetOutput().GetPointData().GetArrayName(ii)
         self.arrayNames.append(name)

      # Print all relavent informations
      print('GRID info, ncells: ', self.ncell, ', npoints: ', self.npoints, 
       ', narray: ', self.narray, ', array names: ', self.arrayNames)


   def importVTK(self, filename, datatype='CELL', overset=False):
      #if isinstance(filename, list):
      self.data = mv.ReadVTKFile(filename)
      if datatype == 'CELL':
         self.datac = mv.ReadVTKFile(filename)
         self.data = mv.CtoP(mv.ReadVTKFile(filename))

      self.ncell=  self.data.GetOutput().GetNumberOfCells()
      self.npoints=self.data.GetOutput().GetNumberOfPoints()
      self.narray =self.data.GetOutput().GetPointData().GetNumberOfArrays()
      self.arrayNames = []

      for ii in range(self.narray):
         name = self.data.GetOutput().GetPointData().GetArrayName(ii)
         self.arrayNames.append(name)

      # Print all relavent informations
      print('GRID info, ncells: ', self.ncell, ', npoints: ', self.npoints, 
       ', narray: ', self.narray, ', array names: ', self.arrayNames)

      #for name in self.caseOptions['solvarnames']:
      #   varname = self.caseOptions['solvarnames'][name]
      #   if varname in self.arrayNames:
      #      self.vars.append(varname)
      for name in self.arrayNames:
         self.vars.append(name)


   def exportVTK(self, filename, varlist=None):
      if varlist:
         self.computeVar(varlist)
      data = self.data
      mv.WriteVTKGridFile(data, filename)
      return

   def makePlot(self, array=None, axlim=None):
      if array is None:
          nrow, ncol = 1, 1 
      else:
          nrow, ncol = array[0], array[1] 

      fig, ax = plt.subplots(nrow, ncol)

      if axlim is not None:
         ax.set_xlim(axlim[0])
         ax.set_ylim(axlim[1])


      return fig, ax


   def savePlot(self):
      pass
      #return fig, ax

   def setDefaultStrings(self):
      # set default latex expressions for plotting
      # with no units
      title_strs =  \
             {'alpha':r'$\alpha$',
             'cl':r'$c_\ell$',
             'cd':r'$c_d$',
             'cmz':r'$c_{m_{c/4}}$',
             'cT':r'$c_T$',
             'cP':r'$c_P$',
             'CL':r'$C_L$',
             'CD':r'$C_D$',
             }
      # with units
      label_strs =  \
             {'alpha':r'$\alpha$ [deg]',
             'cl':r'$c_\ell$',
             'cd':r'$c_d$',
             'cmz':r'$c_{m_{c/4}}$',
             'cT':r'$c_T$',
             'cP':r'$c_P$',
             'CL':r'$C_L$',
             'CD':r'$C_D$',
             }
      return title_strs, label_strs
   
def findBoundaryEdges(edges):
   edgescopy = edges.tolist()
   print(edgescopy[:20])
   for ed in edgescopy:
      ed.sort()
   print(edgescopy[:20])
   edgescopy = list(map(tuple, edgescopy))
   print(len(edgescopy))
   cnt = Counter(edgescopy)
   unique_edges = []
   for edge in set(edgescopy):
      if cnt[edge] == 1:
         unique_edges.append(edge)
   unique_edges.sort()
   del edgescopy

   return  unique_edges  

