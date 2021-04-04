#!/usr/bin/env python
import vtk
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
from scipy.interpolate import griddata
import math
import math as m
import os

import vtk4cfd.mytool as mt


"""
VTK function library

Developer:
-------------------
- Raye Tianbo Xie

"""

## Create unstructured grid
def CreateUGrid(pts, conn, scalars=None, vectors=None, ctype='hex'):
   """
   Function to create a vtk unstructured grid
   (only support hex cells + nodal data for now)

   Parameters:
   ------------
   pts: list of points of dim{Nptsx3}
   conn: list of connectivity dim{Ncellsx8}
   scalars: dictionary of scalar data [name:dim{Nptsx1}]
   vectors: dictionary of vector data [name:dim{Nptsx3}]
   ctype: cell type: 'quad' or 'hex'

   Return:
   -----------
   grid: vtk grid object
   """
   conn = np.asarray(conn)
   npts = len(pts)
   ncells = len(conn)
   points = vtk.vtkPoints()
   for i in range(npts):
      points.InsertPoint(i, pts[i])
   grid = vtk.vtkUnstructuredGrid()
   grid.SetPoints(points)
   grid.Allocate(100)
   # insert cells
   for i in range(ncells):
      if ctype == 'hex':
         grid.InsertNextCell(vtk.VTK_HEXAHEDRON, 8, conn[i,:])
      elif ctype == 'quad':
         grid.InsertNextCell(vtk.VTK_QUAD, 4, conn[i,:])
   # insert each scalar
   if scalars is not None:
      for name, scalar in scalars.items():
         assert(len(scalar) == npts)
         array = vtk.vtkDoubleArray()
         array.SetName(name)
         for value in scalar:
            array.InsertNextValue(value)
         grid.GetPointData().AddArray(array)
   # insert each vector
   if vectors is not None:
      for name, vector in vectors.items():
         assert(len(vector) == npts)
         assert(len(vector[0]) == 3)
         array = vtk.vtkDoubleArray()
         array.SetName(name)
         array.SetNumberOfComponents(3)
         for value in vector:
            array.InsertNextTuple(value)
         grid.GetPointData().AddArray(array)

   return grid

def WriteUGrid(grid, filename):
   writer = vtk.vtkUnstructuredGridWriter()
   writer.SetInputData(grid)
   writer.SetFileName(filename)
   writer.Update()

## Read data from vtk file ##
def ReadVTKFile( filename ):
  reader = vtk.vtkGenericDataObjectReader()
  reader.SetFileName(filename)
  reader.ReadAllScalarsOn()
  reader.ReadAllVectorsOn()
  reader.Update()
  return reader

def ReadTestCase( casename ,
                  zone     ,
		  time     ,
		  dt=None   ):
  TestCaseRoot='/home/tianboxi/SU2/Testcases'
  filepath=TestCaseRoot+'/'+casename+'/'+'flow_'+str(zone)+'_'+time+'.vtk'
  reader = ReadVTKFile(filepath)
  return reader

def CombineZones( zone1      ,
                  zone2      ,
		  zone3=None ,
		  zone4=None  ):
  append = vtk.vtkAppendFilter()
  append.AddInputData(zone1.GetOutput())
  append.AddInputData(zone2.GetOutput())
  if zone3:
    append.AddInputData(zone3.GetOutput())
  if zone4:
    append.AddInputData(zone4.GetOutput())
  append.Update()
  return append

def appendDom( dom , append=None):
  if append is None:
     append = vtk.vtkAppendFilter()
  append.AddInputData(dom.GetOutput())
  append.Update()
  return append

def NToXYZ( n ):
  if n==0:
    exp='X'
  elif n==1:
    exp='Y'
  elif n==2:
    exp='Z'
  return exp

def SetActiveData( source        ,
                   name          ,
		   component=None ):
  array = source.GetOutput().GetPointData().GetArray(name)
  if array.GetNumberOfComponents()>1:
    source = ArrayMag(source,name)
    arraymag = source.GetOutput().GetPointData().GetArray(name+'-Magnitude') 
    if component:
      source = ExtractArrayComponent2(source,name,component)
      source.GetOutput().GetPointData().SetActiveScalars(name+'-'+NToXYZ(component))
    else:
      source.GetOutput().GetPointData().SetActiveScalars(name+'-Magnitude')
    source.GetOutput().GetPointData().SetActiveVectors(name)
  else:
    source.GetOutput().GetPointData().SetActiveScalars(name)
  return source

# Definition of implicit functions
def CreatePlaneFunction(origin,normal):
  plane = vtk.vtkPlane()
  plane.SetOrigin(origin)
  plane.SetNormal(normal)
  return plane

def CreateCylinderFunction(radius,center,rotatex,rotatey):
  cylinder = vtk.vtkCylinder()
  cylinder.SetRadius(radius)
  cylinder.SetCenter(center)
  transf = vtk.vtkTransform()
  transf.RotateX(rotatex)
  transf.RotateY(rotatey)
  cylinder.SetTransform(transf)
  return cylinder

# Definition of sources for probes
def CreatePointSource(center,radius,numberofpoints=5000):
  point = vtk.vtkPointSource()
  point.SetCenter(center)
  point.SetRadius(radius)
  point.SetDistributionToUniform()
  point.SetNumberOfPoints(numberofpoints)
  return point

def CreateLineSource(p1,p2,res):
  line = vtk.vtkLineSource()
  line.SetResolution(res)
  line.SetPoint1(p1)
  line.SetPoint2(p2)
  line.Update()
  return line

def CreateCircleSource(radius,center,res):
  circle = vtk.vtkRegularPolygonSource()
  circle.SetNumberOfSides(res)
  circle.SetRadius(radius)
  circle.SetCenter(center)
  circle.Update()
  return circle

def CreatePlaneSource(origin,p1,p2,res1,res2):
#SetOrigin sets one corner of the plane.
#SetPoint1 and SetPoint2 set the corners adjacent to that one.
  plane = vtk.vtkPlaneSource()
  plane.SetOrigin(origin)
  plane.SetPoint1(p1)
  plane.SetPoint2(p2)
  plane.SetResolution(res1,res2)
  plane.Update()
  return plane

# Clipping the dataset
def Clip(inputdata, clipfunction):
  clipper = vtk.vtkClipDataSet()
  clipper.SetInputData(inputdata.GetOutput())
  clipper.SetClipFunction(clipfunction)
  #clipper.GenerateClipScalarsOn()
  clipper.Update()
  return clipper

#Cutting dataset
def Cut(inputconnection, cutfunction, tri=True):
  cutter = vtk.vtkCutter()
  cutter.SetInputConnection(inputconnection.GetOutputPort())
  cutter.SetGenerateTriangles(tri)
  cutter.SetCutFunction(cutfunction)
  cutter.Update()
  return cutter

#Probing dataset with a source
def Probe(source,probewithsource):
  probe = vtk.vtkProbeFilter()
  probe.SetInputConnection(probewithsource.GetOutputPort())
  probe.SetSourceData(source.GetOutput())
  probe.Update()
  return probe

# Compute gradients
def Grad(source, array_name, result_name):
  gfilter = vtk.vtkGradientFilter()
  gfilter.SetInputData(source)
  # Note the first parameter of SetInputScalars
  # 0 - input array associated with points
  # 1 - input array associated with cells
  gfilter.SetInputScalars(0, array_name)
  gfilter.SetResultArrayName(result_name)
  gfilter.ComputeDivergenceOff()
  gfilter.ComputeVorticityOff()
  gfilter.ComputeQCriterionOff()
  gfilter.Update()
  return gfilter

# Data Conversions
def GetNodes(source):
  nodes = source.GetOutput().GetPoints().GetData()
  return nodes

def GetNodesOnAxialPlane(source, axial_loc):
  N = source.GetOutput().GetNumberOfPoints()
  node_array = np.empty([0,3])
  for i in range(N):
    nodei = source.GetOutput().GetPoint(i)
    nodei_np = np.array([[nodei[0], nodei[1], nodei[2]]])
    if nodei[2] == axial_loc:
      node_array = np.append(node_array,nodei_np,axis=0)
  return node_array
      
def GetArray(source,index):
  fielddata = source.GetOutput().GetPointData().GetArray(index)
  return fielddata
def GetCArray(source,index):
  fielddata = source.GetOutput().GetCellData().GetArray(index)
  return fielddata
def GetScalar(source):
  scalar = source.GetOutput().GetPointData().GetScalars()
  return scalar
  """ Convert cell data to pointdata """
def CtoP(source):
  celltopoint = vtk.vtkCellDataToPointData()
  celltopoint.SetInputConnection(source.GetOutputPort())
  celltopoint.Update()
  return celltopoint
def GetVector(scource):
  vector = scource.GetOutput().GetPointData().GetVectors()
  return vector

def GetCellVolume(source):
  quality = vtk.vtkMeshQuality()
  quality.SetInputData(source.GetOutput())
  quality.SetHexQualityMeasureToVolume()
  quality.Update()
  vol = quality.GetOutput().GetCellData().GetArray('Quality')
  return vol

def GetTriangles(source):
  '''
  Get triangle vertex indices from a vtkPolydata (for example a surface cut)
  '''
  polydata = source.GetOutput()
  ncell = polydata.GetNumberOfCells()
  conn = []
  for i in range(ncell):
     thisCell = polydata.GetCell(i)
     thisInd = []
     for j in range(3):
        thisInd.append(thisCell.GetPointId(j))
     conn.append(thisInd)

  return conn

# Data processing within vtkDataSet
def ArrayMag( source  ,
              name     ):
  mag = vtk.vtkArrayCalculator()
  mag.SetInputConnection(source.GetOutputPort())
  mag.AddVectorArrayName(name)
  mag.SetResultArrayName(name+'-Magnitude')
  mag.SetFunction('mag('+name+')')
  mag.Update()
  return mag
def ExtractArrayComponent( source   ,
                           name     ,
			   component ):
  array = source.GetOutput().GetPointData().GetArray(name)
  newarray = vtk.vtkFloatArray()
  newarray.SetNumberOfComponents(0)
  newarray.SetName(name+'-'+NToXYZ(component))
  for i in range(source.GetOutput().GetNumberOfPoints()):
    newarray.InsertNextValue(array.GetComponent(i,component))
  source.GetOutput().GetPointData().AddArray(newarray)
  return source

def ExtractArrayComponent2( source   ,
                            name     ,
			    component ):
  newsource = vtk.vtkArrayCalculator()
  newsource.SetInputConnection(source.GetOutputPort())
  newsource.AddScalarArrayName(name,component)
  newsource.SetResultArrayName(name+'-'+NToXYZ(component))
  newsource.SetFunction(name)
  newsource.Update()
  return newsource

def AddNewArray(source,array,name):
  N = array.shape[0]
  if len(array.shape)==1:
    n = 1
  else: 
    n = array.shape[1]
  newarray = vtk.vtkFloatArray()
  newarray.SetNumberOfComponents(n)
  newarray.SetNumberOfTuples(N)
  newarray.SetName(name)
  for i in range(N):
    if len(array.shape)==1:
      newarray.SetComponent(i,0,array[i])
    else:
      for j in range(n):
        newarray.SetComponent(i,j,array[i,j])
  source.GetOutput().GetPointData().AddArray(newarray)
  return source

#def CalcQuantities(source):
#  calc = vtk.vtkArrayCalculator()
#  omega = 1200
#  omega = omega*2*math.pi/60
#  calc.SetInputConnection(source.GetOutputPort())
#  calc.AddVectorArrayName('Momentum')
#  calc.AddScalarArrayName('Density')
#  calc.SetResultArrayName('Velocity')
#  calc.SetFunction('Momentum/Density')
#  calc.Update()
#  source = calc
#  nodes = vtk_to_numpy(GetNodes(source))
#  v = vtk_to_numpy(GetArray(source,'Velocity'))
#  N = len(v) 
#  cylnodes = np.empty([N,3])
#  cylnodes[:,2] = nodes[:,2]
#  CylVelocity = np.empty([N,3])
#  CylVelocity[:,2] = v[:,2]
#  RelVelocity = np.empty([N,3])
#  RelVelocity[:,2] = v[:,2]
#  RelCylVelocity = np.empty([N,3])
#  RelCylVelocity[:,2] = v[:,2]
#  for j in range(N):
#    x,y,z = nodes[j,0], nodes[j,1], nodes[j,2]
#    r = math.sqrt(math.pow(x,2)+math.pow(y,2))
#    cylnodes[j,0] = r
#    if y>=0:
#      theta = math.acos(x/r)
#    elif y<0:
#      theta = 2*math.pi - math.acos(x/r)
#    cylnodes[j,1] = theta
#  rary = cylnodes[:,0]
#  vx,vy,vz = v[:,0], v[:,1], v[:,2]
#  vr = np.cos(theta)*vx + np.sin(theta)*vy
#  vt = -1*np.sin(theta)*vx + np.cos(theta)*vy
#  wt = vt + omega*rary
#  wx = np.cos(theta)*vr-np.sin(theta)*wt
#  wy = np.sin(theta)*vr+np.cos(theta)*wt
#  Velocity = v
#  CylVelocity[:,0],CylVelocity[:,1] = vr, vt
#  RelVelocity[:,0],RelVelocity[:,1] = wx, wy
#  RelCylVelocity[:,0],RelCylVelocity[:,1] = vr, wt
#  KineticEng = np.power(vx,2)+np.power(vy,2)+np.power(vz,2)
#  Energy = vtk_to_numpy(GetArray(source,'Energy'))
#  TEnthalpy = Energy+1.0/2*KineticEng 
#  returnlist = [nodes, cylnodes, Velocity, CylVelocity, RelVelocity, RelCylVelocity, TEnthalpy]
#  return returnlist

# Rendering process 
def ColorTransferFun(data,cl=[0.6,0.6,0.6],name=None,component=None):
  #Get range of scalar
  if name:
    array = data.GetOutput().GetPointData().GetArray(name)
    if component:
      Range = array.GetRange(component)
    elif array.GetNumberOfComponents()>1:
      arraymag = data.GetOutput().GetPointData().GetArray(name+'-Magnitude')
      Range = arraymag.GetRange()
    else:
      Range = array.GetRange()
    lower = Range[0]
    upper = Range[1]
    #Building the color transfer function based on range of scalar
    lut =vtk.vtkColorTransferFunction()
    lut.AddRGBPoint(lower, 0.231373, 0.298039, 0.752941)
    lut.AddRGBPoint((lower+upper)/2, 0.865003, 0.865003, 0.865003)
    lut.AddRGBPoint(upper, 0.705882, 0.0156863, 0.14902)
    lut.SetColorSpaceToDiverging()
  else:
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(1)
    lut.SetTableValue(0,cl[0],cl[1],cl[2])
  #Opacity mapping function
  #ofun = vtk.vtkPiecewiseFunction()
  #ofun.AddPoint(lower,0)
  #ofun.AddPoint(upper,0)
  return lut

def CreateScalarBar(cm,name,component=None):
  cbar = vtk.vtkScalarBarActor()
  cbar.SetLookupTable(cm)
  if component:
    cbar.SetTitle(name+'-'+NToXYZ(component))
  else: 
    cbar.SetTitle(name)
  cbar.SetNumberOfLabels(10)
  cbar.SetWidth(0.09)
  cbar.SetHeight(0.5)
  cbar.SetLabelFormat('%-#6.3f')
  cbar.GetLabelTextProperty().SetColor(0,0,0)
  cbar.GetLabelTextProperty().ShadowOff()
  cbar.GetLabelTextProperty().SetFontFamily(1)
  cbar.GetTitleTextProperty().SetFontFamily(1)
  cbar.GetTitleTextProperty().SetColor(0,0,0)
  cbar.GetTitleTextProperty().ShadowOff()
  cbar.SetPosition(0.9,0.1)
  cbar.SetVerticalTitleSeparation(10)
  #TODO Add vtkAxisActor2D aligning with color bar
  return cbar

def CreateDataMapper(source,cm=None,name=None,component=None):
  dmap = vtk.vtkDataSetMapper()
  if cm:
    dmap.SetLookupTable(cm)
  if name:
    array = source.GetOutput().GetPointData().GetArray(name)
    if component:
      dmap.SetInputConnection(source.GetOutputPort())
      Range = array.GetRange(component)
      dmap.SelectColorArray(name)
    elif array.GetNumberOfComponents()==1:
      dmap.SetInputConnection(source.GetOutputPort())
      Range = array.GetRange()
      dmap.SelectColorArray(name)
    else:
      dmap.SetInputConnection(source.GetOutputPort())
      arraymag = source.GetOutput().GetPointData().GetArray(name+'-Magnitude')
      Range = arraymag.GetRange()
    dmap.SetScalarRange(Range)
    dmap.SetUseLookupTableScalarRange(1)
    dmap.ScalarVisibilityOn()
  else:
    dmap.SetInputConnection(source.GetOutputPort())
  return dmap

def CreateActor(mapper):
  #Actor for data
  dactor = vtk.vtkActor()
  dactor.SetMapper(mapper)
  #dactor.GetProperty().SetOpacity(opacity)
  return dactor

def RenderActor(dactor, campos, name=None, component=None):
  #Create Renderer
  ren = vtk.vtkRenderer()
  #Create Render window
  renwin = vtk.vtkRenderWindow()
  renwin.AddRenderer(ren)
  renwin.SetSize( 1000, 1000 )
  #Add Render Window Interactor
  iren = vtk.vtkRenderWindowInteractor()
  iren.SetRenderWindow(renwin)
  #Add orientation axis
  axesactor = vtk.vtkAxesActor()
  axes = vtk.vtkOrientationMarkerWidget()
  axes.SetOrientationMarker(axesactor)
  axes.SetInteractor(iren)
  axes.SetEnabled(1)
  if name:
    cm = dactor[0].GetMapper().GetLookupTable()
    bar = CreateScalarBar(cm,name,component)
    ren.AddActor(bar)
  #Add actors 
  for i in range(len(dactor)):
    ren.AddActor(dactor[i])
  ren.SetBackground(0.34,0.32,0.43)
  #Set the camera to position
  cam = ren.GetActiveCamera()
  SetCamera(cam, campos)
  ren.ResetCamera()
  #Start Rendering
  iren.Initialize()
  renwin.Render()
  iren.Start()
  return 

def SetCamera(cam, campos):
  cam.SetFocalPoint(0,0,0)
  if campos == 'Zp':
    cam.SetPosition(0,0,-1)
    cam.SetViewUp(0,1,0)
  elif campos == 'Zn':
    cam.SetPosition(0,0,1)
    cam.SetViewUp(0,1,0)
  elif campos == 'Xp':
    cam.SetPosition(-1,0,0)
    cam.SetViewUp(0,1,0)
  elif campos == 'Xn':
    cam.SetPosition(1,0,0)
    cam.SetViewUp(0,1,0)
  elif campos == 'Yp':
    cam.SetPosition(0,-1,0)
    cam.SetViewUp(1,0,0)
  elif campos == 'Yn':
    cam.SetPosition(0,1,0)
    cam.SetViewUp(1,0,0)
  cam.ComputeViewPlaneNormal()
  return cam

## --------------------- ##
## My plotting functions ##
## --------------------- ##

def DataActor3D(source,name=None, component=None):
  cl = [0.6,0.6,0.6]
  if name:
    source = SetActiveData(source,name,component)
    ct = ColorTransferFun(source,cl,name,component)
  else:
    ct = ColorTransferFun(source,cl)
  mapper = CreateDataMapper(source,ct,name,component)
  actor = CreateActor(mapper)
  #RenderActor([actor],name,component)
  return actor

def Glyph(source,cl,name,params=[500,0.11],scalebyvectoron=0):
  source.GetOutput().GetPointData().SetActiveVectors(name)
  #Arrow source
  arrow = vtk.vtkArrowSource()
  arrow.SetTipResolution(6)
  arrow.SetTipRadius(0.1)
  arrow.SetTipLength(0.35)
  arrow.SetShaftResolution(6)
  arrow.SetShaftRadius(0.03)
  #Use point mask to subsample the points
  ptmask = vtk.vtkMaskPoints()
  ptmask.SetInputConnection(source.GetOutputPort())
  ptmask.SetOnRatio(params[0])
  ptmask.RandomModeOn()
  #glyph 
  glyph = vtk.vtkGlyph3D()
  glyph.SetInputConnection(ptmask.GetOutputPort())
  glyph.SetSourceConnection(arrow.GetOutputPort())
  glyph.SetVectorModeToUseVector()
  glyph.SetColorModeToColorByVector()
  if scalebyvectoron==1:
    glyph.SetScaleModeToScaleByVector()
  glyph.OrientOn()
  glyph.SetScaleFactor(params[1])
  glyph.Update()
  mapper = vtk.vtkPolyDataMapper()
  mapper.SetInputConnection(glyph.GetOutputPort())
  cm = ColorTransferFun(source,cl)
  mapper.SetLookupTable(cm)
  actor = CreateActor(mapper)
  return actor

def Streamline(source,name,point,srctype='ptsource'):
  source = SetActiveData(source,name)
  tracer = vtk.vtkStreamTracer()
  tracer.SetInputConnection(source.GetOutputPort())
  if srctype == 'ptsource':
    tracer.SetSourceConnection(point.GetOutputPort())
  elif srctype == 'singlept':
    tracer.SetStartPosition(point)
  tracer.SetIntegrationDirectionToBoth()
  tracer.SetIntegratorTypeToRungeKutta45()
  tracer.SetInitialIntegrationStep(0.001)
  tracer.SetMaximumIntegrationStep(0.01)
  tracer.SetMinimumIntegrationStep(0.0001)
  tracer.SetMaximumPropagation(0.4)
  tracer.Update()

  tube = vtk.vtkTubeFilter()
  tube.SetInputConnection(tracer.GetOutputPort())
  tube.SetRadius(0.005)
  tube.CappingOn()
  tube.SetNumberOfSides(6)
  cl = [0.6,0.6,0.6] 
  cm = ColorTransferFun(source,cl,name)
  mapper = vtk.vtkPolyDataMapper()
  mapper.SetInputConnection(tube.GetOutputPort())
  mapper.SelectColorArray(name+'-Magnitude')
  mapper.SetLookupTable(cm)
  actor = CreateActor(mapper)
  return actor

def AxisymmetricContour(source,name,component=0,ax=None,Z=None,X=None,n=None):
  rcParams['contour.negative_linestyle'] = 'solid'
  source = SetActiveData(source,name,component)
  planeyz = CreatePlaneFunction([0,0,6],[1,0,0])
  planexz = CreatePlaneFunction([0,0,6],[0,-1,0])
  clip = Clip(source, planeyz)
  cut = Cut(clip, planexz)
  nodes = vtk_to_numpy(GetNodes(cut))
  data = vtk_to_numpy(GetScalar(cut))
  #if len(array.shape) !=1:
  #  data = array[:,component]
  #else:
  #  data = array
  x,y,z = nodes[:,0], nodes[:,1], nodes[:,2]
  #xmin,xmax = np.amin(x),np.amax(x)
  #zmin,zmax = np.amin(z),np.amax(z)
  if (Z is None):
    xi = np.linspace(xmin,xmax,1000)
    zi = np.linspace(zmin,zmax,1000)
    X, Z = np.meshgrid(xi,zi)
  #xmin,xmax = np.amin(X),np.amax(X)
  #zmin,zmax = np.amin(Z),np.amax(Z)
  #linearly interpolate from x,z,pressure to meshgrid X,Z
  pi = griddata((x,z), data, (X,Z), method='linear')
  #fig = plt.figure()
  #ax = fig.add_subplot(1,1,1)
  cs = ax.contourf(Z, X, pi, 100)
  css = ax.contour(Z, X, pi, 20, colors='k') # solid line contour
  ax.clabel(css, fontsize=10, inline=1)
  if (n):
    ax.plot(Z[n[0],:],X[n[0],:],'k-')
    ax.plot(Z[n[1],:],X[n[1],:],'k-')
    ax.plot(Z[:,0],X[:,0],'k-')
    ax.plot(Z[:,-1],X[:,-1],'k-')
  ax.set_aspect('equal')
  ax.set_xlim([-0.5,1.5])
  #ax.set_ylim([xmin,xmax])
  #ax.set_xticks(np.linspace(6,6.3,3))
  #cbar = fig.colorbar(cs)
  #cbar.set_label(name)
  #plt.show()
  ax.set_xlabel('Normalized Z')
  ax.set_ylabel('Normalized R')
  return cs



def AxialContour(source,name,zloc,nlvl=50):
  source.GetOutput().GetPointData().SetActiveScalars(name)
  plane = CreatePlaneFunction((0,0,zloc),(0,0,1))
  cut = Cut(source,plane)
  nodes_vtk = GetNodes(cut)
  pressure_vtk = GetScalar(cut)
  nodes = vtk_to_numpy(nodes_vtk)
  nodes_cyl = mt.CartToCyl(nodes,'coord')
  x,y,z = nodes[:,0], nodes[:,1], nodes[:,2]
  xmin, xmax = min(x), max(x)
  ymin, ymax = min(y), max(y)
  rmin, rmax = min(nodes_cyl[:,0]), max(nodes_cyl[:,0])
  ri = np.linspace(rmin,rmax,1000)
  ti = np.linspace(0, m.pi*2,1000)
  X,Y = np.empty([1000,1000]),np.empty([1000,1000])
  for j in range(1000):
   for i in range(1000):
     X[i,j] = ri[i]*m.cos(ti[j])
     Y[i,j] = ri[i]*m.sin(ti[j])
  pressure = vtk_to_numpy(pressure_vtk)
  #linearly interpolate from x,z,pressure to meshgrid X,Z
  pi = griddata((x,y), pressure, (X,Y), method='linear')
  fig = plt.figure()
  ax = fig.add_subplot(1,1,1)
  cs = ax.contourf(X, Y, pi, nlvl)
  ax.set_aspect('equal')
  ax.set_xlim([xmin,xmax])
  ax.set_ylim([ymin,ymax])
  cbar = fig.colorbar(cs)
  cbar.set_label(name)
  plt.axis('off')
  plt.show()
  return

def AxialProbes(source, name, numberofprobes, z1,z2,rlocs = None):
  source.GetOutput().GetPointData().SetActiveScalars(name)
  n = numberofprobes
  if rlocs==None:
     rlocs=np.zeros(n)
     for i in range(n):
       rlocs[i] = i*(float(1)/(n-1))*(1.3-0.5)+0.5
  fig = plt.figure()
  ax = fig.add_subplot(1,1,1)
  for i in range(n):
     if n>1:
       r = rlocs[i] 
     else:
       r = rlocs
     xy = r/math.sqrt(2)
     line = CreateLineSource([xy,xy,z1],[xy,xy,z2],100)
     probe = Probe(source,line)
     nodes_vtk = GetNodes(probe)
     pressure_vtk = GetScalar(probe)
     nodes = vtk_to_numpy(nodes_vtk)
     x,y,z = nodes[:,0], nodes[:,1], nodes[:,2]
     #xmin, xmax = min(x), max(x)
     #ymin, ymax = min(y), max(y)
     zmin, zmax = min(z), max(z)
     pressure = vtk_to_numpy(pressure_vtk)
     z = np.delete(z,[100])
     pressure = np.delete(pressure,[100])
     line, =ax.plot(z,pressure, label='r='+str(r))
     ax.set_xlim([zmin,zmax])
  lgend = plt.legend(loc=4,labelspacing=0.2)
  plt.xlabel('Axial location')
  plt.ylabel(name)
  plt.grid(True)
  plt.show()
  return

def RadialProbe(source,name,r1,r2,zloc,component=0,ax=None):
  source = SetActiveData(source,name,component)
  data = np.zeros((201))
  for j in range(10):
    theta = 2*m.pi/10*j
    x1,x2 = r1*m.cos(theta),r2*m.cos(theta)
    y1,y2 = r1*m.sin(theta),r2*m.sin(theta)
    line = CreateLineSource([x1,y1,zloc],[x2,y2,zloc],200)
    probe = Probe(source,line)
    nodes = vtk_to_numpy(GetNodes(probe))
    nodes_cyl = mt.CartToCyl(nodes,'coord')
    data = data + vtk_to_numpy(GetScalar(probe))
    #if len(array.shape)!=1:
    #  data = data + array[:,component]
    #else:
    #  data = data + array
  data = data/10
  for i in range(len(data)):
    if data[i] !=0:
      ind1 = i
      break
  for i in range(len(data)-ind1-1):
    if data[i+ind1] == 0:
      ind2 = ind1+i
      break
  #fig,ax = plt.subplots(1,1)
  ax.plot(data[ind1:ind2],nodes_cyl[ind1:ind2,0])
  ax.set_ylabel('Normalized R')
  #ax.set_xlabel(name)
  #plt.grid(True)
  #plt.show()
  return
#def CircProbes()

def MassAverage(source,name,zloc):
  ntan,nsec = 30,20 
  source = SetActiveData(source,name)
  plane = CreatePlaneFunction((0,0,zloc),(0,0,1))
  cut = Cut(source,plane)
  nodes = vtk_to_numpy(GetNodes(cut))
  rho = vtk_to_numpy(GetArray(cut,'Density'))
  v = vtk_to_numpy(GetArray(cut,'Velocity'))
  vz = v[:,2]
  data = vtk_to_numpy(GetScalar(cut))
  nodes_cyl = mt.CartToCyl(nodes,'coord')
  x,y = nodes[:,0], nodes[:,1] 
  xmin, xmax = min(x), max(x)
  ymin, ymax = min(y), max(y)
  rmin, rmax = min(nodes_cyl[:,0]), max(nodes_cyl[:,0])
  #rmin = m.ceil(rmin*100000)/100000
  #rmax = m.floor(rmax*100000)/100000
  ri = np.linspace(rmin,rmax,nsec+1)
  ti = np.linspace(0, m.pi*2,ntan)
  X,Y = np.empty([ntan,nsec]),np.empty([ntan,nsec])
  for j in range(ntan):
    for i in range(nsec):
      X[j,i] = ri[i]*m.cos(ti[j])
      Y[j,i] = ri[i]*m.sin(ti[j])
  #linearly interpolate from x,z,pressure to meshgrid X,Z
  datai = griddata((x,y), data, (X,Y), method='linear')
  rhoi = griddata((x,y), rho, (X,Y), method='linear')
  vi = griddata((x,y), vz, (X,Y), method='linear')
  numerator,mdot = 0.0,0.0
  mt.WriteData([datai],'data/datai.dat','R')
  for j in range(ntan):
    for i in range(nsec-1):
      if j != ntan-1:
        p1 = [X[j,i],Y[j,i]]
        p2 = [X[j,i+1],Y[j,i+1]]
        p3 = [X[j+1,i],Y[j+1,i]]
        p4 = [X[j+1,i+1],Y[j+1,i+1]]
        Area = mt.shoelace4(p1,p2,p4,p3)
        dataave = (datai[j,i]+datai[j,i+1]+datai[j+1,i]+datai[j+1,i+1])/4
        rhoave = (rhoi[j,i]+rhoi[j,i+1]+rhoi[j+1,i]+rhoi[j+1,i+1])/4
        vave = (vi[j,i]+vi[j,i+1]+vi[j+1,i]+vi[j+1,i+1])/4
        numerator = numerator + dataave*rhoave*vave*Area
        mdot = mdot + rhoave*vave*Area 
      else:
        p1 = [X[j,i],Y[j,i]]
        p2 = [X[j,i+1],Y[j,i+1]]
        p3 = [X[0,i],Y[0,i]]
        p4 = [X[0,i+1],Y[0,i+1]]
        Area = mt.shoelace4(p1,p2,p4,p3)
        dataave = (datai[j,i]+datai[j,i+1]+datai[0,i]+datai[0,i+1])/4
        rhoave = (rhoi[j,i]+rhoi[j,i+1]+rhoi[0,i]+rhoi[0,i+1])/4
        vave = (vi[j,i]+vi[j,i+1]+vi[0,i]+vi[0,i+1])/4
        numerator = numerator + dataave*rhoave*vave*Area
        mdot = mdot + rhoave*vave*Area 
  massave = numerator/mdot 
  return massave,mdot

# Output vtk file  
def WriteVTK(X,Y,Z,name,datatype='surface',datalist=None):
  if datatype =='surface':
    npts = X.shape[0] 
    nsec = X.shape[1]
    n = X.size
    ncell = (npts-1)*(nsec-1) 
    outputfile = open(name,'w')
    outputfile.write("# vtk DataFile Version 3.0\n")
    outputfile.write(name)
    outputfile.write("\nASCII\n")
    outputfile.write("DATASET UNSTRUCTURED_GRID\n")
    outputfile.write("POINTS %i double\n" % n)
    for j in range(nsec):
      for i in range(npts):
        outputfile.write("%f %f %f\n" % (X[i,j],Y[i,j],Z[i,j]))
    outputfile.write("CELLS %i %i\n" % (ncell, (ncell*5)))
    for j in range(nsec-1):
      for i in range(npts-1):
        outputfile.write("%i %i %i %i %i\n" % (4, npts*j+i, npts*j+i+1, npts*(j+1)+i+1, npts*(j+1)+i))
    outputfile.write("CELL_TYPES %i\n" % ncell)
    for i in range(ncell):
      outputfile.write("%i\n" % 9)
    outputfile.close()
  elif datatype =='volume':
    npts = X.shape[0] 
    nsec = X.shape[1]
    ntan = X.shape[2]
    ntotal = X.size
    ncell = (npts-1)*(nsec-1)*ntan 
    outputfile = open(name,'w')
    outputfile.write("# vtk DataFile Version 3.0\n")
    outputfile.write(name)
    outputfile.write("\nASCII\n")
    outputfile.write("DATASET UNSTRUCTURED_GRID\n")
    outputfile.write("POINTS %i double\n" % ntotal)
    count = 0
    ind = np.empty([npts,nsec,ntan])
    for k in range(ntan):
      for j in range(nsec):
        for i in range(npts):
          outputfile.write("%f %f %f\n" % (X[i,j,k],Y[i,j,k],Z[i,j,k]))
          ind[i,j,k] = count
          count=count+1
    outputfile.write("CELLS %i %i\n" % (ncell, (ncell*9)))
    for k in range(ntan):
      for j in range(nsec-1):
        for i in range(npts-1):
          elem1 = ind[i,j,k]
          elem2 = ind[i+1,j,k]
          elem3 = ind[i+1,j+1,k]
          elem4 = ind[i,j+1,k]
          if k == ntan-1:
            elem5 = ind[i,j,0] 
            elem6 = ind[i+1,j,0] 
            elem7 = ind[i+1,j+1,0] 
            elem8 = ind[i,j+1,0] 
          else:
            elem5 = ind[i,j,k+1] 
            elem6 = ind[i+1,j,k+1] 
            elem7 = ind[i+1,j+1,k+1] 
            elem8 = ind[i,j+1,k+1] 
    outputfile.write("%i %i %i %i %i %i %i %i %i\n" % (8,elem1,elem2,elem3,elem4,elem5,elem6,elem7,elem8))
    outputfile.write("CELL_TYPES %i\n" % ncell)
    for i in range(ncell):
      outputfile.write("%i\n" % 12)
    if datalist:
      outputfile.write('POINT_DATA %i\n' % (ntotal))
      ndata = len(datalist)
      for n in range(ndata):
        if len(datalist[n][0].shape) == 3: # Scalar data
          entry = datalist[n][0]
          name = datalist[n][1]
          outputfile.write('SCALARS ')
          outputfile.write(name)
          outputfile.write(' double 1\nLOOKUP_TABLE default\n')
          for k in range(ntan):
              for j in range(nsec):
                for i in range(npts):
                  outputfile.write("%f \t" % (entry[i,j,k]))
          outputfile.write('\n')
        elif len(datalist[n][0].shape) == 4: # Vector data
          entry = datalist[n][0]
          name = datalist[n][1]
          outputfile.write('VECTORS ')
          outputfile.write(name)
          outputfile.write(' double\n')
          for k in range(ntan):
            for j in range(nsec):
              for i in range(npts):
                for l in range(3):
                  outputfile.write("%f " % (entry[i,j,k,l]))
                outputfile.write('\t')
    outputfile.close()

def ConvertToPoly(data):
   polyfilter = vtk.vtkGeometryFilter()
   polyfilter.SetInputData(data.GetOutput())
   polyfilter.Update()
   return polyfilter
def WriteVTKFile(data,filename):
   writer = vtk.vtkPolyDataWriter()
   writer.SetInputData(data.GetOutput())
   writer.SetFileName(filename)
   writer.Update()
def WriteVTKGridFile(data,filename):
   writer = vtk.vtkUnstructuredGridWriter()
   writer.SetInputData(data.GetOutput())
   writer.SetFileName(filename)
   writer.Update()

   return

