import nurbspy as nrb
import numpy as np
import matplotlib.pyplot as plt

# define the list of points to fit to
points = np.array([[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)],
                   [(0.0, 1.0, 0.0), (1.0, 1.0, 0.0), (2.0, 1.0, 0.0)],
                   [(0.0, 2.0, 0.0), (1.0, 2.0, 0.0), (2.0, 2.0, 0.0)]
])

# P is the control points that we modify
# intitialize P as flat surface to start.
P = np.array([[(0.0, 0.0, 0.0), (1.0, 0.0, 2.0), (2.0, 0.0, 0.0)],
              [(0.0, 1.0, 2.0), (1.0, 1.0, -2.0), (2.0, 1.0, 2.0)],
              [(0.0, 2.0, 0.0), (1.0, 2.0, 2.0), (2.0, 2.0, 0.0)]
])
# transpose to get in proper format
points = np.array(points).transpose((2, 1, 0))
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

    # se = squared error
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
    # mse = mean of squared error
    return mse

control_points = P

# # set learning rate and number of iterations
# learning_rate = 0.01
# num_iterations = 40

# # perform gradient descent
# for i in range(num_iterations):
#     # compute gradient of loss function w.r.t. control points
#     gradient = np.zeros_like(control_points)
#     for j in range(control_points.shape[0]):
#         for k in range(control_points.shape[1]):
#             for l in range(control_points.shape[2]):
#                 delta = 1e-5
#                 control_points_plus = np.copy(control_points)
#                 control_points_plus[j,k,l] += delta
#                 loss_plus = loss_function(control_points_plus)
#                 control_points_minus = np.copy(control_points)
#                 control_points_minus[j,k,l] -= delta
#                 loss_minus = loss_function(control_points_minus)
#                 gradient[j,k,l] = (loss_plus - loss_minus) / (2 * delta)

#     # update control points based on gradient and learning rate
#     control_points -= learning_rate * gradient

#     # print loss every 10 iterations
#     if i % 2 == 0:
#         print(f"Iteration {i}: Loss {loss_function(control_points)}")

final_surface = nrb.NurbsSurface(control_points)

# plot the final surface with the control points
final_surface.plot(control_points=True, isocurves_u=8, isocurves_v=8)
plt.show()
