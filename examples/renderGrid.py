import numpy as np
import vtk4cfd as mv

data = mv.ReadVTKFile('./inputFiles/surface.vtk')
act = mv.DataActor3D(data)
mv.RenderActor([act],'Yn')
