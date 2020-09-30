import vtk4cfd as mv
import numpy as np
import json
from vtk.util.numpy_support import vtk_to_numpy

# exp of methods
#read = mv.ReadVTKFile('volume.vtk')
#print(read.GetOutput().GetPointData().GetArray('Normals'))
pts = [[0,0,0],[0,0,1],[0,1,1],[0,1,0],
       [1,0,0],[1,0,1],[1,1,1],[1,1,0]]
conn = [[0,1,2,3,4,5,6,7]]
pts1 = [[0,0,0],[0,0,1],[0,1,1],[0,1,0]]
conn1 = [[0,1,2,3]]
       
scalar1 = [1,3,5,3,6,2,6,7]
scalar2 = [2,3,5,3,1,2,6,7]
vector1 = [[1,3,3],[1,1,1],[2,2,2],[3,3,3],[4,4,4],[2,4,6],[2,3,6],[7,3,2]]
# scalar and vector as dictionary 
scalars = { 'sc1':scalar1, 'sc2':scalar2 }
vectors = { 've1':vector1 }

grid_data = { 'pts':pts, 'conn':conn, 
              'sca':scalars, 'vec':vectors }

# dump into json object
grid_json = json.dumps(grid_data)

# dump into json file
with open("exp_data_file.json", "w") as write_file:
   json.dump(grid_data, write_file)

# read from json file
with open("exp_data_file.json", "r") as read_file:
   data = json.load(read_file)

#print(data['sca']['sc1'])

# vtk unstructured grid object
grid = mv.CreateUGrid(pts, conn, scalars=scalars, vectors=vectors, ctype='hex')
grad = mv.Grad(grid, 'sc1', 'grad')
grid = grad.GetOutput()
print(vtk_to_numpy(grid.GetPointData().GetArray('ve1')))

# surface grid 
grid1 = mv.CreateUGrid(pts1, conn1, ctype='quad')
# export VTK file
mv.WriteUGrid(grid,'testgridvol.vtk')
mv.WriteUGrid(grid1,'testgridsurf.vtk')
