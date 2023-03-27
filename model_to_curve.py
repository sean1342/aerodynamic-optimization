import nurbspy as nrb
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# define the list of points to fit to
points = np.array([[(1.5, 1.5, 0.0), (1.0, 0.0, 2.0), (2.0, 0.0, 0.0)],
                   [(0.0, 1.0, 2.0), (1.0, 1.0, 0.0), (2.0, 1.0, 2.0)],
                   [(0.0, 2.0, 0.0), (1.0, 2.0, 1.0), (2.0, 2.0, 0.0)]
])

# P is the control points that we modify
# initialize P to the points to fit to to start off
P = np.array([[(0.0, 0.0, 0.0), (1.0, 0.0, 2.0), (2.0, 0.0, 0.0)],
              [(0.0, 1.0, 2.0), (1.0, 1.0, -2.0), (2.0, 1.0, 2.0)],
              [(0.0, 2.0, 0.0), (1.0, 2.0, 2.0), (2.0, 2.0, 0.0)]
])
# transpose to get in proper format
P = np.array(P).transpose((2, 1, 0))

# create the NURBS surface
surface = nrb.NurbsSurface(control_points=P)

# utility function for 3d distance
def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)

# define loss function
# loss = sum(distances between points and points on surface) **
def loss_function(control_points):
    # reshape control_points to match the original shape
    control_points = np.reshape(control_points, P.shape)
    surface = nrb.NurbsSurface(control_points)

    coords = []
    resolution = 50
    for i in range(resolution):
        for j in range(resolution):
            u,v = i/50, j/50
            coords.append(surface.get_value(u, v).flatten())

    se = []
    for _ in points:
        for point in _:
            # TODO: fix this bad solution
            best_coord = [9999,9999,9999]
            best_dist = distance(point, best_coord)
            for coord in coords:
                if (distance(point, coord) < best_dist).any():
                    best_coord = coord
                    best_dist = distance(point, coord)

            # print(point)
            # print(best_coord)
            se.append(best_dist ** 2)
    mse = np.mean(se)
    print(mse)

    return mse

initial_guess = P.flatten()

# gradient descent here

# plot the final surface with the control points
surface.plot(control_points=True, isocurves_u=8, isocurves_v=8)
plt.show()
