import math as m
from math import pow
from math import sqrt
from math import acos
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import optimize
from mpl_toolkits.mplot3d import Axes3D
import csv
# Global vars
c0,c1 = 0.0,0.0
# Scale a list of numpy arraies by a factor
def ScaleArray(datalist, factor):
  ndata = len(datalist)
  for i in range(ndata):
    datalist[i] = datalist[i]*factor
  return datalist
def NormalizeArray(data):
  N = data.shape[0]
  n = data.shape[1]
  data_n = np.empty([N,n])
  for i in range(N):
    l2norm = 0.0
    for j in range(n):
      l2norm = l2norm+data[i,j]**2 
    l2norm = sqrt(l2norm)
    if l2norm != 0:
      for j in range(n):
        data_n[i,j] = data[i,j]/l2norm
    else: 
      for j in range(n):
        data_n[i,j] = 0.0
  return data_n
def NormalizeVector(data):
  n = len(data)
  data_n = np.empty([n])
  l2norm = 0.0
  for j in range(n):
    l2norm = l2norm+data[j]**2
  l2norm = sqrt(l2norm)
  for j in range(n):
    data_n[j] = data[j]/l2norm
def DotProd(data1,data2):
  N = data1.shape[0]
  data_n = np.empty([N,1])
  for i in range(N):
    data_n[i] = data1[i,0]*data2[i,0] + data1[i,1]*data2[i,1] + data1[i,2]*data2[i,2]
  return data_n

def Norm(data):
  n = len(data)
  summ = 0
  for i in range(n):
    summ = summ + data[i]**2
  return sqrt(summ)
  
  
# Convert cartesian coords or vector to cylindrical coords or vector
def CartToCyl(data, dtype='coord',theta=None):
  N = data.shape[0]
  n = data.shape[1]
  output = np.empty([N,n])
  if dtype == 'coord':
    output[:,0] = np.sqrt(data[:,1]**2+data[:,2]**2)
    for i in range(N):
      if data[i,2]>=0:
        output[i,1] = acos(data[i,1]/output[i,0])
      else:
        output[i,1] = m.pi*2 - acos(data[i,1]/output[i,0])
    if n == 3: #3D coord
      output[:,2] = data[:,0]
  elif dtype == 'vector':
    output[:,0] =    np.cos(theta[:])*data[:,1] + np.sin(theta[:])*data[:,2]
    output[:,1] = -1*np.sin(theta[:])*data[:,1] + np.cos(theta[:])*data[:,2]
    if n == 3: #3D vector
      output[:,2] = data[:,0]
  return output
# Convert cylindrical coords or vector to cartesian coords or vector
def CylToCart(data,dtype='coord', theta=None):
  N = data.shape[0]
  n = data.shape[1]
  output = np.empty([N,n])
  if dtype == 'coord':
    output[:,1] = data[:,0] * np.cos(data[:,1])
    output[:,2] = data[:,0] * np.sin(data[:,1])
    if n == 3:
      output[:,0] = data[:,2]
  elif dtype == 'vector':
    output[:,1] = np.cos(theta[:])*data[:,0] - np.sin(theta[:])*data[:,1]
    output[:,2] = np.sin(theta[:])*data[:,0] + np.cos(theta[:])*data[:,1]
    if n == 3:
      output[:,0] = data[:,2]
  return output

# Calculate point distribution for clustering
def Cluster(startpos,spacing_ini, length, npoints, direction):
  global c0,c1
  if (npoints-1)*spacing_ini <= length:
    c0 = length/spacing_ini
    nspace = npoints -1
    c1 = nspace 
    k = optimize.brentq(fun, 1.00001, 10)
    Z = np.empty([npoints])
    if direction == 'left':
      Z[0] = startpos
      for i in range(nspace):
        Z[i+1] = Z[i] + spacing_ini*(k**i)
      Z[npoints-1] = Z[0]+length
    elif direction == 'right':
      Z[npoints-1] = startpos
      for i in range(nspace):
        Z[npoints-1-(i+1)] = Z[npoints-1-i] - spacing_ini*(k**i)
      Z[0] = Z[npoints-1] - length
    else:
      print("Error:cluster direction is not specified\n")
  else:
    print("Error: reduce #points or initial spcing")
    if direction == 'left':
      Z = np.linspace(startpos,startpos+length,npoints)
    elif direction == 'right':
      Z = np.linspace(startpos-length,startpos,npoints)
  return Z
def fun(k):
  global c0,c1
  return (k**c1-c0*k+(c0-1))

# function that joins 2D curves
def JoinPath(pathlist):
  nlist = len(pathlist)
  n = pathlist[0].shape[1]
  joined = np.empty([0,n])
  for i in range(nlist):
    joined = np.append(joined,pathlist[i],axis=0)
  return joined

# Create a divergent duct shape
def Divfun(x,y0):
  n = len(x)
  y = np.empty([n])
  for i in range(n):
    if x[i]-x[0]<8.43789:
      y[i] = (x[i]-x[0])**2/6 + y0
    elif x[i]-x[0]>=8.43789:
      y[i] = np.arctan((x[i]-6.17-x[0])/2)*14 + y0
  return y
# Adjust the 2D mesh for the radial location of streamwise lines
def AdjustStreamwiselines(Rtarget,R,loc):
  nsec =  R.shape[1]
  npath = R.shape[0] 
  Znew,Rnew = np.zeros([npath,nsec]),np.zeros([npath,nsec])
  if loc =='left':
    Radialmisloc = Rtarget -  R[0,:]
    for i in range(npath):
      coeff = pow(float(i-npath+1)/(npath-1),2)
      R[i,:] = R[i,:] + Radialmisloc*coeff
  elif loc =='right':
    Radialmisloc = Rtarget - R[npath-1,:]
    for i in range(npath):
      coeff = pow(float(i)/(npath-1),2)
      R[i,:] = R[i,:] + Radialmisloc*coeff
  return R

  
# Ajust Quasi-Orthogonal lines for curvatures
def AdjustQOlines(Ztarget,Rtarget,Z,R,loc):
  nsec =  Z.shape[1]
  npath = Z.shape[0] 
  Znew,Rnew = np.zeros([npath,nsec]),np.zeros([npath,nsec])
  A = np.array([Ztarget[nsec-1],Rtarget[nsec-1]])
  B = np.array([Ztarget[0],Rtarget[0]])
  dist = np.empty([npath,nsec])
  if loc =='left':
    dist[0,0],dist[0,nsec-1] = 0,0
    for i in range(nsec-2): # excluding the hub and shroud point 
      C = np.array([Ztarget[i+1],Rtarget[i+1]]) 
      dist[0,i+1] = SolveDist(A,B,C) 
    for i in range(npath): 
      coeff = pow(float(i-npath+1)/(npath-1),2)
      dist[i,:] = dist[0,:]*coeff 
  elif loc =='right':
    dist[npath-1,0],dist[npath-1,nsec-1] = 0,0
    for i in range(nsec-2): # excluding the hub and shroud point 
      C = np.array([Ztarget[i+1],Rtarget[i+1]]) 
      dist[npath-1,i+1] = SolveDist(A,B,C) 
    for i in range(npath): 
      coeff = pow(float(i)/(npath-1),2)
      dist[i,:] = dist[npath-1,:]*coeff 
  Znew[:,0],Rnew[:,0]= Z[:,0],R[:,0]
  Znew[:,nsec-1],Rnew[:,nsec-1] = Z[:,nsec-1],R[:,nsec-1]
  for j in range(nsec-2):
    for i in range(npath):
      A = np.array([Z[i,nsec-1],R[i,nsec-1]])
      D = np.array([Z[i,j+1],R[i,j+1]])
      C = SolvePtC(A,D,dist[i,j+1]) 
      Znew[i,j+1] = C[0]
      Rnew[i,j+1] = C[1]
  return Znew,Rnew

#Function that solves for the distance from point C to line AB
def SolveDist(A,B,C):
  AC = np.array([C[0]-A[0], C[1]-A[1]])
  AB = np.array([B[0]-A[0], B[1]-A[1]])
  dist = sqrt(pow(AC[0],2)+pow(AC[1],2)-pow(AC[0]*AB[0]+AC[1]*AB[1],2)/(pow(AB[0],2)+pow(AB[1],2)))
  if AB[1]*AC[0]-AB[0]*AC[1]>0: #cross product
    dist = -dist
  return dist
#Function thet sovles for point C knowing A,D and dist
def SolvePtC(A,D,dist):
  ACp = np.array([0.1,0.1])
  AC = np.array([0.1,0.1])
  C = np.array([0.1,0.1])

  AD = np.array([D[0]-A[0], D[1]-A[1]])
  absAD = sqrt(pow(AD[0],2)+pow(AD[1],2))
  ACp[0] = absAD
  ACp[1] = dist
  tt = m.acos(AD[0]/absAD)
  AC[0] = m.cos(tt)*ACp[0] + m.sin(tt)*ACp[1]
  AC[1] =-m.sin(tt)*ACp[0] + m.cos(tt)*ACp[1] 
  C[0] = AC[0]+A[0]
  C[1] = AC[1]+A[1]
  return C
# given origin radius start and end angle return circ arc
def Circ(origin,r,t1,t2,pts=30):
  coord = np.empty([pts,2])
  dt = (t2-t1)/(pts-1)
  for i in range(pts): 
    coord[i,0] = origin[0]+r*m.cos(t1+dt*i)
    coord[i,1] = origin[1]+r*m.sin(t1+dt*i)
  return coord
# function that write data of the same len in to columns
def WriteCSV(datalist,name):
   ncol = len(datalist)
   outfile = open(name,'w')
   lencol = len(datalist[0])
   for i in range(lencol):
     for j in range(ncol):
       outfile.write('%f, ' % datalist[j][i])
     outfile.write('\n')
   outfile.close()

#def WriteData(datalist,name,dtype='3Dvector'):
#  if dtype == '3Dvector':
#    ndata = len(datalist)
#    nlen = datalist[0].size
#    nx = datalist[0].shape[0]
#    ny = datalist[0].shape[1]
#    nz = datalist[0].shape[2]
#    nk = datalist[0].shape[3]
#    outfile = open(name,'w')
#    counter = 1
#    for k in range(nz):
#      for j in range(ny):
#        for i in range(nx):
#          #outfile.write('%i ' % (counter))
#	  counter = counter+1
#	  for l in range(ndata):
#	    for m in range(nk):
#	      entry = datalist[l][i,j,k,m]
#	      outfile.write('%.3f ' % (entry))
#	  outfile.write('\n')
#    outfile.close()
#  if dtype == 'Meridional':
#    ndata = len(datalist)
#    nx = datalist[0].shape[0]
#    ny = datalist[0].shape[1]
#    nk = [0]*ndata
#    for i in range(ndata):
#      if len(datalist[i].shape) == 3:
#        nk[i] = datalist[i].shape[2]
#    outfile = open(name,'w')
#    outfile.write('NPTS=%i\n' % nx)
#    outfile.write('NSEC=%i\n' % ny)
#    for j in range(ny):
#      for i in range(nx):
#	      for l in range(ndata):
#	        nnk = nk[l]
#	        if nnk !=0:
#	          for m in range(nnk):
#	            entry = datalist[l][i,j,m]
#	            outfile.write('%f ' % (entry))
#	        else:
#	            entry = datalist[l][i,j]
#	            outfile.write('%f ' % (entry))
#	        outfile.write('\n')
#    outfile.close()   
#  if dtype == 'R':
#    ndata = len(datalist)
#    N = datalist[0].shape[0]
#    outfile = open(name,'w')
#    for i in range(N):
#      for j in range(ndata):
#        if len(datalist[j].shape)!=1:
#          for k in range(datalist[j].shape[1]):
#            outfile.write('%f ' % datalist[j][i,k])
#	      else:
#	        outfile.write('%f ' % datalist[j][i])
#      outfile.write('\n')
#    outfile.close()
#  if dtype == 'CSV':
#    ncol = len(datalist)
#    outfile = open(name,'w')
#    lencol = len(datalist[0])
#    for i in range(lencol):
#      for j in range(ncol):
#        outfile.write('%f, ' % datalist[j][i])
#      outfile.write('\n')
#    outfile.close()

# Output su2 mesh file   
def Writesu2(X,Y,Z,name,blext=None,nblext=None):  
  npts = X.shape[0] 
  nsec = X.shape[1]
  ntan = X.shape[2]
  ntotal = X.size
  if blext is None:
    nblext = 1
  ncell = (npts-1)*(nsec-1)*ntan 
  count = 0
  ind = np.empty([npts,nsec,ntan])
  for k in range(ntan):
    for j in range(nsec):
      for i in range(npts):
        ind[i,j,k] = count
        count=count+1
  outputfile = open(name,'w')
  outputfile.write('NDIME=%i\n' % 3)
  outputfile.write('NELEM=%i\n' % ncell)
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
        outputfile.write("%i %i %i %i %i %i %i %i %i\n" % (12,elem1,elem2,elem3,elem4,elem5,elem6,elem7,elem8))
  outputfile.write('NPOIN=%i\n' % ntotal)
  for k in range(ntan):
    for j in range(nsec):
      for i in range(npts):
        outputfile.write("%f %f %f\n" % (X[i,j,k],Y[i,j,k],Z[i,j,k]))
  if blext:
    outputfile.write('NMARK=%i\n' % 6)
    extncell       = (nblext-1)*ntan
    sidewallncell = (npts-nblext)*ntan
  else:
    outputfile.write('NMARK=%i\n' % 4)
    sidewallncell = (npts-1)*ntan
  capncell = (nsec-1)*ntan
  # Buid inner wall 
  outputfile.write('MARKER_TAG=inner_wall\n')
  outputfile.write('MARKER_ELEMS=%i\n' % sidewallncell)
  for k in range(ntan):
    for i in range(npts-1-(nblext-1)):
      i = i + (nblext-1)
      j = 0 
      if k==ntan-1:
        outputfile.write('%i %i %i %i %i\n' % (9,ind[i,j,k],ind[i+1,j,k],ind[i+1,j,0],ind[i,j,0]))
      else:
        outputfile.write('%i %i %i %i %i\n' % (9,ind[i,j,k],ind[i+1,j,k],ind[i+1,j,k+1],ind[i,j,k+1]))
  # Build outer wall
  outputfile.write('MARKER_TAG=outer_wall\n')
  outputfile.write('MARKER_ELEMS=%i\n' % sidewallncell)
  for k in range(ntan):
    for i in range(npts-1-(nblext-1)):
      i = i + (nblext-1)
      j = nsec-1 
      if k==ntan-1:
        outputfile.write('%i %i %i %i %i\n' % (9,ind[i,j,k],ind[i+1,j,k],ind[i+1,j,0],ind[i,j,0]))
      else:
        outputfile.write('%i %i %i %i %i\n' % (9,ind[i,j,k],ind[i+1,j,k],ind[i+1,j,k+1],ind[i,j,k+1]))
  if blext:
    # Build inner extension wall
    outputfile.write('MARKER_TAG=inner_wall_ext\n')
    outputfile.write('MARKER_ELEMS=%i\n' % extncell)
    for k in range(ntan):
      for i in range(nblext-1):
        j = nsec-1 
        if k==ntan-1:
          outputfile.write('%i %i %i %i %i\n' % (9,ind[i,j,k],ind[i+1,j,k],ind[i+1,j,0],ind[i,j,0]))
        else:
          outputfile.write('%i %i %i %i %i\n' % (9,ind[i,j,k],ind[i+1,j,k],ind[i+1,j,k+1],ind[i,j,k+1]))
    # Build outer extension wall
    outputfile.write('MARKER_TAG=outer_wall_ext\n')
    outputfile.write('MARKER_ELEMS=%i\n' % extncell)
    for k in range(ntan):
      for i in range(nblext-1):
        j = nsec-1 
        if k==ntan-1:
          outputfile.write('%i %i %i %i %i\n' % (9,ind[i,j,k],ind[i+1,j,k],ind[i+1,j,0],ind[i,j,0]))
        else:
          outputfile.write('%i %i %i %i %i\n' % (9,ind[i,j,k],ind[i+1,j,k],ind[i+1,j,k+1],ind[i,j,k+1]))
  # Build inlet cap 
  outputfile.write('MARKER_TAG=inlet\n')
  outputfile.write('MARKER_ELEMS=%i\n' % capncell)
  for k in range(ntan):
    for j in range(nsec-1):
      i = 0 
      if k==ntan-1:
        outputfile.write('%i %i %i %i %i\n' % (9,ind[i,j,k],ind[i,j+1,k],ind[i,j+1,0],ind[i,j,0]))
      else:
        outputfile.write('%i %i %i %i %i\n' % (9,ind[i,j,k],ind[i,j+1,k],ind[i,j+1,k+1],ind[i,j,k+1]))
  # Build outlet cap
  outputfile.write('MARKER_TAG=outlet\n')
  outputfile.write('MARKER_ELEMS=%i\n' % capncell)
  for k in range(ntan):
    for j in range(nsec-1):
      i = npts-1 
      if k==ntan-1:
        outputfile.write('%i %i %i %i %i\n' % (9,ind[i,j,k],ind[i,j+1,k],ind[i,j+1,0],ind[i,j,0]))
      else:
        outputfile.write('%i %i %i %i %i\n' % (9,ind[i,j,k],ind[i,j+1,k],ind[i,j+1,k+1],ind[i,j,k+1]))
def ReadMerCoord(name,mertype):
  Z = np.empty([0,1])
  R = np.empty([0,1])
  with open(name) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=' ')
    for row in csv_reader: 
       Z = np.append(Z,np.array([[float(row[0])]]),axis=0)
       R = np.append(R,np.array([[float(row[1])]]),axis=0)
  csv_file.close() 
  if mertype == 'Euler':
    Z = np.reshape(Z,(188,100),order='F')
    R = np.reshape(R,(188,100),order='F')
  elif mertype == 'NS':
    Z = np.reshape(Z,(178,200),order='F')
    R = np.reshape(R,(178,200),order='F')
  return Z,R

""" 
read a specific column from a CSV file for a range of rows, index start from 0
""" 
def readCSV(fileName,col,fromrow,utlrow=None,delim=','):
   if utlrow is None:
       utlrow = -1
       data = np.empty([0])
   elif utlrow-fromrow>1:
       data = np.empty([0])
       array = 1
   else:
       data = 0.0
       array = 0
   with open(fileName) as csvfile:
       reader = csv.reader(csvfile, delimiter=delim)
       rowcount = 0 
       for row in reader:
           if rowcount>=fromrow and row[col]:
               if utlrow==-1:
                   data = np.append(data, float(row[col]))
               else:
                   if rowcount>utlrow:
                       break
                   elif array:
                       data = np.append(data, float(row[col]))
                   else:
                       data = float(row[col])
           rowcount = rowcount + 1
   return data
def ReadCSVCol(filename,col,skip):
  data = np.empty([0])
  with open(filename) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    count = 0
    for row in csv_reader:
      if count>=skip and row[col]:
        data = np.append(data,float(row[col]))
      count = count + 1
  csv_file.close()
  return data

def shoelace3(p1,p2,p3):      
  A = 0.5*m.fabs(p1[0]*p2[1]+p2[0]*p3[1]+p3[0]*p1[1]-p2[0]*p1[1]-p3[0]*p2[1]-p1[0]*p3[1])
  return A
 
def shoelace4(p1,p2,p3,p4):      
  A = 0.5*m.fabs(p1[0]*p2[1]+p2[0]*p3[1]+p3[0]*p4[1]+p4[0]*p1[1]-p2[0]*p1[1]-p3[0]*p2[1]-p4[0]*p3[1]-p1[0]*p4[1])
  return A
