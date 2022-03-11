#!/usr/bin/env python
#from mypostprocessing import TCase
import numpy as np
import csv
from math import acos
# ================================================================
# Coordinates Transformations 
# ================================================================
def transform(data, rot_center, rot_angs, trans):
   """
   tansform a set of corrdinates by applying translating to a new origin, and rotation around 3 axis 
   --
   data: size Nx3 array
   rot_center: size 3 array
   rot_angs: size 3, list, rotation angles in degrees around x, y and z axis
      theta = rot_angs[0] axis rotation around x axis
      phi = rot_angs[1] axis rotation around y axis
      psi = rot_angs[2] axis rotation around z axis
   """
   assert(data.shape[1] == 3)
   assert(len(data.shape) ==2)
   rot_center = np.asarray(rot_center)
   trans = np.asarray(trans)

   cos_1, sin_1 = np.cos(rot_angs[0]), np.sin(rot_angs[0])
   cos_2, sin_2 = np.cos(rot_angs[1]), np.sin(rot_angs[1])
   cos_3, sin_3 = np.cos(rot_angs[2]), np.sin(rot_angs[2])

   rot_matrix = np.zeros((3,3))

   rot_matrix[0,0] = cos_2*cos_3
   rot_matrix[0,1] = sin_1*sin_2*cos_3 - cos_1*sin_3
   rot_matrix[0,2] = cos_1*sin_2*cos_3 + sin_1*sin_3

   rot_matrix[1,0] = cos_2*sin_3
   rot_matrix[1,1] = sin_1*sin_2*sin_3 + cos_1*cos_3
   rot_matrix[1,2] = cos_1*sin_2*sin_3 - sin_1*cos_3

   rot_matrix[2,0] = -1*sin_2
   rot_matrix[2,1] = sin_1*cos_2
   rot_matrix[2,2] = cos_1*cos_2  

   data = data - rot_center 
   # [Nx3] = ([3x3][3xN]).T
   transformed_data = (rot_matrix@data.T).T

   transformed_data = transformed_data + rot_center + trans

   return transformed_data

def xyzTortz(coords, axis='x'):
   """
   convert cartesian coordinate to cylindrical system about given axis
   --
   coords, N by 3 double array, N points, coordinate in 3 dirs
   """
   coords = np.asarray(coords)
   if axis == 'x':
      rr = np.sqrt(coords[:,1]**2 + coords[:,2]**2) # sqrt(y^2+z^2)
      tt = np.arctan2(coords[:,2], coords[:,1]) # atan(z/y)
      xx = coords[:,0] 
   result = np.asarray([rr, tt, xx])
   return result.T

def rtzToxyz(coords, axis='x'):
   """
   convert cylindrical coordinates to cartesian about given axis 
   --
   coords, N by 3 double array, N points, coordinate in 3 dirs
   """
   coords = np.asarray(coords)
   if axis == 'x':
      xx = coords[:,2]
      yy = coords[:,0]*np.cos(coords[:,1])
      zz = coords[:,0]*np.sin(coords[:,1])
   result = np.asarray([xx,yy,zz])
   return result.T

def rotate(data, ax, ang):
   """
   rotate vectors about axis ccw by ang 

   ax: int list[3], example [1,0,0] for x-axis
   ang: numpy.pi for 180 degs
   """
   assert(data.shape[1] == 3)
   assert(len(data.shape) ==2)

   ux,uy,uz = ax[0], ax[1], ax[2]
   sin,cos = np.sin(ang), np.cos(ang) 
   omcos = 1-np.cos(ang)
   # define rotational matrix
   rot_max =np.asarray([[cos + ux**2*omcos  , ux*uy*omcos - uz*sin, ux*uz*omcos + uy*sin],\
                        [uy*ux*omcos + uz*sin, cos + uy**2*omcos   , uy*uz*omcos - ux*sin],\
                        [uz*ux*omcos - uy*sin, uz*uy*omcos + ux*sin, cos + uz**2*omcos   ]])
   # Apply rotation rot_max{3x3} * data{3xN} = rot_data{3xN}
   rot_data = np.dot(rot_max,data.T).T

   return rot_data
def vecCyltoCart():
   pass
def vecCarttoCyl():
   pass
# ================================================================
# Misc functions 
# ================================================================
def cind(icase, dsize):
   """ This is a utility function
   given dimension and size of each dimention
   return the location of ith element
   eg. ndim = 2, matrix dsize = [3,4]
   i = 5 -> [2,1] in the matrix
   note: icase and ind are zero based index
   """
   ndim = len(dsize)
   ntot = np.prod(dsize)
   dsizes = [1]
   for i in range(ndim):
      # start from the last element
      dsizes.append(dsize[ndim-1-i]*dsizes[-1])

   # remove the leading element in dsizes (which is 1)
   del dsizes[-1]
   # reverse list order
   dsizes.reverse()
   ind = []
   for i in range(ndim):
      ii = icase//dsizes[i]
      ind.append(ii)
      icase = icase - ii*dsizes[i]
   return ind

def append_new_line(file_name, text_to_append):
   """Append given text as a new line at the end of file"""
   # Open the file in append & read mode ('a+')
   with open(file_name, "a+") as file_object:
      # Move read cursor to the start of file.
      file_object.seek(0)
      # If file is not empty then append '\n'
      data = file_object.read(100)
      if len(data) > 0:
         file_object.write("\n")
      # Append text at the end of file
      file_object.write(text_to_append)

#def ppinfo(name):
#   """ call post processing procedures to get massflow"""
#   result = TCase(filename=name, ndomain=1)
#   val,mas = result.massAvePlane(['P','Pt'])
#   mass_flow_rate = mas[0]
#   return mass_flow_rate


# Migrated from mytool

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

"""
   Reads plot 3D file for a structrued grid (.x)
"""
def readPlot3D(filename):
   #filename = sys.argv[1]
   f = open(filename, 'r')
   x, y, z = [], [], []
   for nline, line in enumerate(f):
      if nline == 0:
         nblk = float(str.split(line)[0])
      if nline == 1:
         # 
         ni = int(str.split(line)[0])
         nj = int(str.split(line)[1])
         nk = int(str.split(line)[2])
         # 4 coordinates per line in .x file
         # Find out how many lines to read for each component
         coord_nline = (ni*nj*nk)//4.0
         if (ni*nj*nk)%4.0 > 0:
            coord_nline = coord_nline + 1
         # range of coords
         rx = [2, coord_nline + 1]
         ry = [coord_nline + 2, coord_nline*2 + 1]
         rz = [coord_nline * 2 + 2, coord_nline * 3 + 1]
      if nline >= 2:
         line_splitted = str.split(line)
         nnum_to_read = len(str.split(line))
         for n in range(nnum_to_read):
            if nline >= rx[0] and nline <= rx[1]:  
               x.append(float(line_splitted[n]))
            if nline >= ry[0] and nline <= ry[1]:  
               y.append(float(line_splitted[n]))
            if nline >= rz[0] and nline <= rz[1]:  
               z.append(float(line_splitted[n]))
   
   # print('number of blocks: ', nblk)
   # print('ni, nj, nk: ', ni, nj, nk)
   # print('nx, ny, nz: ', len(x), len(y), len(z))

   return np.asarray([x,y,z]).T, [ni, nj, nk]
   
   

# Convert cartesian coords or vector to cylindrical coords or vector
def CartToCyl(data, dtype='coord',theta=None):
  N = data.shape[0]
  n = data.shape[1]
  output = np.empty([N,n])
  if dtype == 'coord':
    output[:,0] = np.sqrt(data[:,1]**2+data[:,2]**2)
    output[:,1] = np.arctan2(data[:,2],data[:,1])
    #for i in range(N):
      #if data[i,2]>=0:
      #  output[i,1] = acos(data[i,1]/output[i,0])
      #else:
      #  output[i,1] = np.pi*2 - acos(data[i,1]/output[i,0])
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

def readFile(filename,nvar,splitby=None):
   f = open(filename, 'r')
   allvar = []
   for i in range(nvar):
      allvar.append([])
   for line in f:
      try:
         for i in range(nvar):
            if splitby:
               allvar[i].append(float(str.split(line,splitby)[i]))
            else:
               allvar[i].append(float(str.split(line)[i]))
      except:
         continue
   return allvar

