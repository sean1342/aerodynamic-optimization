# import packages
import numpy as np
import nurbspy as nrb
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from stl import mesh

# define the list of control points
P = [[(1.5, 1.5, 0.0), (1.0, 0.0, 2.0), (2.0, 0.0, 0.0)],
     [(0.0, 1.0, 2.0), (1.0, 1.0, 0.0), (2.0, 1.0, 2.0)],
     [(0.0, 2.0, 0.0), (1.0, 2.0, 1.0), (2.0, 2.0, 0.0)]
]

P = np.array(P).transpose((2, 1, 0))

# create and plot the Bezier surface
surface = nrb.NurbsSurface(control_points=P)
surface.plot(control_points=True, isocurves_u=8, isocurves_v=8)
plt.show()

# set u and v resolution
u_res = 50
v_res = 50

# create u and v arrays
u = np.linspace(0, 1, u_res)
v = np.linspace(0, 1, v_res)

# evaluate the surface at the u, v values
points = np.zeros((u_res, v_res, 3))
for i in range(u_res):
    for j in range(v_res):
        points[i,j,:] = surface.get_value(u[i], v[j]).reshape(3,)

# print the coordinates of the first 5 points
# print(points[:5])

# create the mesh object
vertices = points.reshape(-1, 3)
faces = Delaunay(vertices[:, :2]).simplices
surface_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
for i, f in enumerate(faces):
    for j in range(3):
        surface_mesh.vectors[i][j] = vertices[f[j], :]

# save the mesh to an STL file
surface_mesh.save('/home/sean/dev/aerodynamic-optimizer/surface.stl')