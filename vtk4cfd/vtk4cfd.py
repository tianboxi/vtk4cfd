import numpy as np
import vtk4cfd.myvtk as mv
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.tri as tri
from mpl_toolkits.axes_grid1 import make_axes_locatable
from vtk.util.numpy_support import vtk_to_numpy
from scipy.interpolate import griddata
import math
import math as m
import os
import vtk


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
      if 'overset' in options:
         overset = options['overset']
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
      copt['freestreamloc'] = None
      copt['solvarnames'] = {'rho':'Density', 'P':'Pressure', 'T':'Temperature',
                             'V':'Velocity', 'M':'Mach', 
                             'rhoet':'EnergyStagnationDensity'} 
      for op in options:
         if op in copt:
            copt[op] = options[op]
      return copt

   def setDefaultPlotOptions(self, options):
      popt = {}
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
      else: 
         pRef = 1.0
         TRef = 1.0
         rhoRef = 1.0

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
      print('INF states: ', infState)
      return infState

   ## ===== Plotting functions ========= ##
   
   def plotContour(self, varname, point=None, normal=None, surface=None, ax=None, 
                   axlim=None, plotboundary=False, cline=None, clevels=None, 
                   cbar=True, clabel=None):

      if surface is not None:
         cut = self.cuts[surface]['vtkcut']
         triangulation = self.cuts[surface]['triangulation']
      elif point is not None and normal is not None:
         cut, triangulation = self.makeSlice(point, normal, self.data) 
      else:
         raise ValueError('not enough input for the contour plot')

      vartoplot = vtk_to_numpy(mv.GetArray(cut, varname))

      if not ax:
         fig, ax = self.makePlot(axlim=axlim)

      if cline is not None:
         nline = 30
         cs = ax.tricontour(triangulation, vartoplot, nline, colors='k')
         ax.clabel(cs, fontsize=6, inline=1)

      if clevels is not None:
         vmin = clevels[0]
         vmax = clevels[1]
         levels = np.linspace(vmin,vmax,50)
         print(name+':'+'SET min='+str(minval)+', max='+str(maxval))
      else:
         levels = 50

      csf = ax.tricontourf(triangulation, vartoplot, 
                        levels=levels, cmap='coolwarm',extend='both')

      if cbar:
         # create an axes on the right side of ax. The width of cax will be 3%
         # of ax and the padding between cax and ax will be fixed at 0.05 inch.
         divider = make_axes_locatable(ax)
         cax = divider.append_axes("right", size="3%", pad=0.05)
         # set color bar, extendrect=true to avoid pointy ends
         cbar = fig.colorbar(csf, cax=cax, orientation='vertical', label=clabel,extendrect=True)
      if plotboundary:
         self.getMeshBoundaries(cut, ax)

      ax.set_aspect('equal')

      plt.show() 
   
   def plotStreamLine(self, point=None, normal=None, surface=None, ax=None, 
                   plotboundary=False):
      pass

   def plotGridLines(self, point=None, normal=None, surface=None, ax=None):
      """
      plot poly grids 
      """
      pass

   def makeSlice(self, point, normal, slicename=None, source=None):
      """
      make a slice cut of the flow domain  
      """
      if source is None:
         source = self.data
      plane = mv.CreatePlaneFunction(point,normal)
      cut = mv.Cut(source,plane)
      # Get triangles from vtk cut 
      triangles = np.asarray(mv.GetTriangles(cut))
      # Reproduce triangulation with matplotlib
      nodes = vtk_to_numpy(mv.GetNodes(cut))
      triangulation = tri.Triangulation(nodes[:,0], nodes[:,1], triangles)
      # save cut for later use if slicename is specified
      if slicename:
         self.cuts[slicename] = {'vtkcut':cut, 'triangulation':triangulation}

      return cut, triangulation

   ## ====== Geometry handeling ====== ##
   def plotAlphaShapes(self, surface, ax=None):
      """
      An implementation of alpha shape through a surface with existing triangulations
      (Could also be done via Delaunay Triangulation)
      See: https://stackoverflow.com/questions/23073170/calculate-bounding-polygon-of-alpha-shape-from-the-delaunay-triangulation

      Input: 
      ---
      surface, a vtk object (same as that of a cut plane) 
      ax, matplotlib object
      """
      print("See getMeshEdges() for similar functionality")
      raise NotImplementedError

   def getMeshBoundaries(self, surface, ax):
      """
      get the boundary curves of a mesh (vtkFeatureEdges)
      """
      edges = mv.GetEdges(surface)
      print(edges.GetOutput().GetLines())
      #nodes = vtk_to_numpy(mv.GetNodes(surface))
      #ptidex = vtk_to_numpy(edges.GetOutput().GetLines().GetData())
      #edgepts = []
      #for ind in ptidex:
      #   edgepts.append(nodes[ptidex, :])
      #print(edgepts)
      #npt = len(edgepts)
      #for ii in range(npt):
      #   ax.plot(edgepts[ii][0], edgepts[ii][1], 'k.')
      


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


   def volInt():
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
         print('You have to specify domain names in the case options thruough: domlist')
         return False

      self.caseOptions['nblk']= len(domlist)
      self.data_doms = []
      # loop through cgns doms
      for dom in domlist:
         self.data_doms.append(mv.ReadVTKFile(dom+'.vtk'))
         # remove converted VTK file
         mycmd = 'rm '+dom+'.vtk'
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


   def importVTK(self, filename, datatype, overset):
      if isinstance(filename, list):
         self.data = mv.ReadVTKFile(dom+'.vtk')
         #TODO: finish testing 

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

      if axlim:
         ax.set_xlim(axlim[0])
         ax.set_ylim(axlim[1])


      return fig, ax


   def savePlot(self):
      pass
      #return fig, ax
   
   

