# VTK4CFD
VTK4CFD is a package designed to post-process CFD results and produce publishable flow visualization plots with simple lines of codes.
This code is built based on python interface of the visualization tool kit (VTK) and plotting library matplotlib. 
The author, Tianbo (Raye) Xie, is a PhD candidate at USC and he used this code to produce result plots for his thesis and publications.
Contact: tianboxi@usc.edu

## Supported File Formats
- CGNS, VTK

## Supported Functions
- Slice domain (plannar or cylindrical)
- Clip domain
- Transform domain
- Flow property integration over a slice surface
- Mass averaged integration over a slice surface
- Flow property sample over an arbitrary curve
- Computation of new variables with existing variables
- Export processed result as a new VTK file

## Supported Plotting Functions
- Flow property contours 
- Streamline (with Overset grid)
- Glyph/vector
- Boundary edges

## Example Plots
![M_chk](https://user-images.githubusercontent.com/32691862/196487391-ba7e11e8-1cd8-4f6e-9459-df41e95e41d1.png)
<img src="https://user-images.githubusercontent.com/32691862/196489901-6623a2a1-f084-41c5-aa67-0f84963d6a25.png" width="500">
![V_far0 075_f30 0_a8 0_BFM](https://user-images.githubusercontent.com/32691862/196490099-9fb699fd-78d7-4ec1-858e-8c64fdc27e81.png)
![Figure_1 (1)](https://user-images.githubusercontent.com/32691862/196490198-24453f81-9298-416e-a554-b6946f5cbe9b.png)


## LICENSE 
Copyright 2020 Tianbo Raye Xie. See LICENCE file for more details. 
