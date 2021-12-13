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
      self.plotOptions = self.setDefaultPlotOptions(plotoptions)
      # this is where main data file is linked to
      self.data = None
      self.drange = {}
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
      # variable existed in obj 
      self.vars = []
      # reference state 
      self.refs = self.setRefState()
      # probe freestream properties
      if self.caseOptions['freestreamloc'] is not None:
         self.infs = self.getFreeStreamProperties(datatype='POINT')     
      # range of data
      self.updateDRange()

   ## ===== Initialization Functions ===== ##

   def setDefaultOptions(self, options):
      copt = {}
      copt['overset'] = False
      copt['refstate'] = None 
      copt['domlist'] = []
      copt['domname_key'] = ['bw02_*.vtk'] 
      copt['freestreamloc'] = None
      copt['solvarnames'] = {'rho':'Density', 'P':'Pressure', 'T':'Temperature',
                             'V':'Velocity', 'M':'Mach', 
                             'rhoet':'EnergyStagnationDensity'} 
      for op in options:
         if op in copt:
            copt[op] = options[op]
      return copt

   def setDefaultPlotOptions(self, options):
      popt = {
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
         if op in popt:
            popt[op] = options[op]
      return popt

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
         rho = vtk_to_numpy(mv.GetArray(self.data,self.caseOptions['solvarnames']['rho']))*rhoRef
         v = vtk_to_numpy(mv.GetArray(self.data,self.caseOptions['solvarnames']['V']))*uRef
         rhoet = vtk_to_numpy(mv.GetArray(self.data,self.caseOptions['solvarnames']['rhoet']))*rhoRef*uRef**2
         P = vtk_to_numpy(mv.GetArray(self.data,self.caseOptions['solvarnames']['P']))*pRef
         M = vtk_to_numpy(mv.GetArray(self.data,self.caseOptions['solvarnames']['M']))

         self.data = mv.AddNewArray(self.data, rho, 'rho')
         self.data = mv.AddNewArray(self.data, P, 'P')
         self.data = mv.AddNewArray(self.data, v, 'V')
         self.data = mv.AddNewArray(self.data, M, 'M')
         self.data = mv.AddNewArray(self.data, rhoet, 'rhoet')
         self.vars = self.vars + ['rho', 'P', 'V', 'M', 'rhoet']

         self.data = mv.RemovePArray(self.data, self.caseOptions['solvarnames']['rho'])
         self.data = mv.RemovePArray(self.data, self.caseOptions['solvarnames']['P'])
         self.data = mv.RemovePArray(self.data, self.caseOptions['solvarnames']['V'])
         self.data = mv.RemovePArray(self.data, self.caseOptions['solvarnames']['M'])
         self.data = mv.RemovePArray(self.data, self.caseOptions['solvarnames']['rhoet'])
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
      print('Data Range: ', self.drange)
         
   def getFreeStreamProperties(self, datatype='POINT',
                               propertyNames=['P','V','rho','M']):
      """
      Probe a spcific point for freestream properties
      """
      loc = self.caseOptions['freestreamloc']
      if datatype=='POINT':
         src = self.data
      else:
         src = self.datac

      infState = self.probFlowField([loc], ['P','V','rho','M'], source=src)
      infState['V'] =  [np.linalg.norm(infState['V'])]
      for state in infState:
         infState[state] = infState[state][0]
      infState['Pt'] =  infState['P']/((1+(1.4-1)/2*infState['M']**2)**(-1.4/(1.4-1)))
      infState['T'] = infState['P']/(287.0*infState['rho'])
      print('INF states: ', infState)
      return infState

   ## ===== Plotting functions ========= ##
   
   def plotContour(self, varname, surface, ax=None, 
                   cbar=True, clabel=None,
                   axlim=None, cline=None, clevels=None):

      cut = self.cuts[surface]['vtkcut']
      triangulation = self.cuts[surface]['triangulation']
      normal = self.cuts[surface]['normal']

      vartoplot = vtk_to_numpy(mv.GetArray(cut, varname))

      if not ax:
         fig, ax = self.makePlot(axlim=axlim)
      else:
         fig = ax.get_figure()

      if cline is not None:
         nline = self.plotOptions['clevels']
         color = self.plotOptions['clcolors']
         cs = ax.tricontour(triangulation, vartoplot, nline, colors='k')
         ax.clabel(cs, fontsize=6, inline=1)

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
            cax = divider.append_axes("right", size=self.plotOptions['cbar_size'], 
            pad=self.plotOptions['cbar_pad'])
            # set color bar, extendrect=true to avoid pointy ends
            cbar = fig.colorbar(csf, cax=cax, orientation='vertical', 
                   label=clabel,extendrect=True)
         elif self.plotOptions['cbar_loc'] == 'bottom':
            cax = divider.append_axes("bottom", size=self.plotOptions['cbar_size'], 
            pad=self.plotOptions['cbar_pad'])
            cbar = fig.colorbar(csf, cax=cax, orientation='horizontal', 
                   label=clabel,extendrect=True)

      ax.set_aspect('equal')

   
   def plotStreamlineMultiBlk(self, varname, points, x_range, 
                           surface=None, ax=None, 
                           idir='both', slname=None, view=None):
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
         
      #nodes = vtk_to_numpy(mv.GetNodes(source))
      npt = len(points)

      allslines = []
      for i, thisPt in enumerate(points):
         sline = mv.Streamline(source, varname, thisPt, idir=idir, planar=planar)
         endx = sline[-1,0]
         counter = 0
         if idir == 'forward':
            while endx < x_range[1]:
               pt_next = sline[-1,:] + (sline[-1,:] - sline[-2,:])*2.0
               sline_next = mv.Streamline(source, varname, pt_next, idir='forward', planar=planar)
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
               sline_next = mv.Streamline(self.data, varname, pt_next, idir='backward', planar=planar)
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

      if slname:
         self.slines[slname] = allslines

      return 


   

   def makeSlice(self, center, normal, slicename=None, source=None, 
                 view='-z', flip=False, saveboundary=False, triangulize=True):
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
      if view == '-x':
         x, y = nodes[:,1], nodes[:,2]
      elif view == '+x':
         x, y = -1*nodes[:,1], nodes[:,2]
      elif view == '-y':
         x, y = -1*nodes[:,0], nodes[:,2]
      elif view == '+y':
         x, y = nodes[:,0], nodes[:,2]
      elif view == '-z':
         x, y = nodes[:,0], nodes[:,1]
      elif view == '+z':
         x, y = -1*nodes[:,0], nodes[:,1]
      if flip:
         xplot, yplot = y, x
      else:
         xplot, yplot = x, y 
      # Convert 3D nodes to 2D coord system on the cut surface
      triangulation = tri.Triangulation(xplot, yplot, triangles)
      # save cut for later use if slicename is specified
      if slicename:
         self.cuts[slicename] = \
         {'vtkcut':cut, 'triangulation':triangulation, 'view':view, 'normal':normal}

      return cut, triangulation

   def makeClip(self, clip_limit, center, normal, clipname=None, source=None):
      """
      make a clip
      ---
      input param:
      
      clip_limit: list, [xmin, xmax, ymin, ymax]
      """
      clip_plane = []
      clip_plane.append(mv.CreatePlaneFunction((clip_limit[0],0.0,0.5),(1,0,0)))
      clip_plane.append(mv.CreatePlaneFunction((clip_limit[1],0.0,0.5),(-1,0,0)))
      clip_plane.append(mv.CreatePlaneFunction((0.0,clip_limit[2],0.5),(0,1,0)))
      clip_plane.append(mv.CreatePlaneFunction((0.0,clip_limit[3],0.5),(0,-1,0)))
      data = self.data
      for clip in clip_plane:
         data = mv.Clip(data, clip)

      self.clips[clipname] = data

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


   def getMeshBoundaries(self, surface, ax=None, name=None):
      """
      get the boundary curves of a mesh (vtkFeatureEdges)
      """
      surface = self.cuts[surface]['vtkcut']
      edge_pts = mv.GetFeatureEdges(surface)
      if name is not None:
         self.outlines[name] = edge_pts
      if ax is not None:
         for edge in edge_pts:
            x = [edge[0][0], edge[1][0]]
            y = [edge[0][1], edge[1][1]]
            ax.plot(x, y, '-k')

   def plotSavedOutlines(self, name, ax): 
      for edge in self.outlines[name]:
         x = [edge[0][0], edge[1][0]]
         y = [edge[0][1], edge[1][1]]
         ax.plot(x, y, self.plotOptions['outline_spec']['spec'],
                 linewidth=self.plotOptions['outline_spec']['w'])

   def plotSavedStreamline(self, name, ax): 
      slines = self.slines[name]
      for sline in slines:
         ax.plot(sline[:-1,0],sline[:-1,1], 
         self.plotOptions['sline_spec']['spec'],
         linewidth=self.plotOptions['sline_spec']['w'])

   def plotGridLines(self,triangulation,ax):
      ax.triplot(triangulation)


   ## ====== CFD data processing ======= ##
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


   def computeVar(self, varlist):
      """
      Given the basic flow solutions, compute other flow variables 
      """
      # import known states as numpy array
      data = self.data
      nodes = vtk_to_numpy(mv.GetNodes(data))

      rho = vtk_to_numpy(mv.GetArray(data,'rho'))
      v = vtk_to_numpy(mv.GetArray(data,'V'))
      rhoet = vtk_to_numpy(mv.GetArray(data,'rhoet'))
      P = vtk_to_numpy(mv.GetArray(data,'P'))
      mach = vtk_to_numpy(mv.GetArray(data,'M'))
      #T = vtk_to_numpy(mv.GetArray(data,'Temperature'))*self.TRef
      T = P/(rho*287.0)
      Et = rhoet/rho
      Ht = Et+P/rho
      Pt = P/((1+(1.4-1)/2*mach**2)**(-1.4/(1.4-1)))    # total pressure from isentropic relation
      N = len(rho)
      
      if 'vx' in varlist and 'vx' not in self.vars:
         data = mv.AddNewArray(data,v[:,0],'vx')      
         self.vars.append('vx')
      if 'vy' in varlist and 'vy' not in self.vars:
         data = mv.AddNewArray(data,v[:,1],'vy')
         self.vars.append('vy')
      if 'vz' in varlist and 'vz' not in self.vars:
         data = mv.AddNewArray(data,v[:,2],'vz')
         self.vars.append('vz')
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
      if 'Pt' in varlist and 'Pt' not in self.vars:
         data = mv.AddNewArray(data,Pt,'Pt')      
         self.vars.append('Pt')
      if 'Tt' in varlist and 'Tt' not in self.vars:
         Tt = T/((1+(1.4-1)/2*mach**2)**(-1))              # total temperature
         data = mv.AddNewArray(data,Tt,'Tt')      
         self.vars.append('Tt')
      if 's' in varlist and 's' not in self.vars:
         S = 287*(1/(1.4-1)*np.log(T)+np.log(1/rho))        # Entropy
         data = mv.AddNewArray(data,S,'s')      
         self.vars.append('s')
      if 'V/Vinf' in varlist and 'V/Vinf' not in self.vars:
         vmag = np.linalg.norm(v, axis=1)
         vovervinf = vmag/self.infs['V']
         data = mv.AddNewArray(data,vovervinf,'V/Vinf')      
         self.vars.append('V/Vinf')
      if 'Cp' in varlist and 'Cp' not in self.vars:
         cp = (P-self.infs['P'])/(0.5*self.infs['rho']*self.infs['V']**2)
         data = mv.AddNewArray(data,cp,'Cp')      
         self.vars.append('Cp')
      if 'Cpt' in varlist and 'Cpt' not in self.vars :
         cpt = (Pt-self.infs['P'])/(0.5*self.infs['rho']*self.infs['V']**2)
         data = mv.AddNewArray(data,cpt,'Cpt')      
         self.vars.append('Cpt')
      if 'Vmag' in varlist and 'Vmag' not in self.vars:
         vmag = np.linalg.norm(v, axis=1)
         data = mv.AddNewArray(data,vmag,'Vmag')      
         self.vars.append('Vmag')

      self.updateDRange()

      return data

   def convertToRotationalFrame(self, center, axis, rpm):
      """
      Convert flow variables from stationary reference to a rotational frame
      """
      pass

   
   ## ====== Integrations ====== ##
   def massInt():
      """
      Mass integration and mass averaging of other properties over a specified surface 
      """
      pass
   
   def surfaceInt():
      """
      Surface integration of flow properties
      """
      pass


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

