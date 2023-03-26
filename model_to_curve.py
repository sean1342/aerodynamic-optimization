import nurbspy as nrb
import numpy as np
import matplotlib.pyplot as plt

# define the list of points to fit to
points = np.array([[(0.0, 0.0, 0.0), (1.0, 0.0, 1.0), (2.0, 0.0, 0.0)],
                   [(0.0, 1.0, 1.0), (1.0, 1.0, 0.0), (2.0, 1.0, 1.0)],
                   [(0.0, 2.0, 0.0), (1.0, 2.0, 1.0), (2.0, 2.0, 0.0)]
])

# P is the control points that we modify
# initialize P to the points to fit to to start off
P = points

# set up the knot vectors for the U and V directions
u_knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
v_knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])

# transpose to get in proper format
P = np.array(P).transpose((2, 1, 0))

# create the NURBS surface
surface = nrb.NurbsSurface(control_points=P, u_knots=u_knots, v_knots=v_knots)

# set up the parameterization of the points to fit to
# using the chord-length parameterization algorithm
num_points = points.shape[0] * points.shape[1]
u_vals = np.zeros(num_points)
v_vals = np.zeros(num_points)
distances = np.zeros(num_points)

for i in range(points.shape[0]):
    for j in range(points.shape[1]):
        idx = i * points.shape[1] + j
        u_vals[idx] = i / (points.shape[0] - 1)
        v_vals[idx] = j / (points.shape[1] - 1)

        if idx > 0:
            distances[idx] = np.linalg.norm(points[i, j] - points[i, j-1])
        if idx < num_points - 1 and j < points.shape[1] - 1:  # fix here
            distances[idx] += np.linalg.norm(points[i, j] - points[i, j+1])


# define the learning rate and number of iterations
learning_rate = 0.1
num_iterations = 100

# perform gradient descent to minimize the distance
for iteration in range(num_iterations):
    # compute the current distances
    current_distances = np.zeros(num_points)
    for idx in range(num_points):
        current_distances[idx] = surface.distance_point(u_vals[idx], v_vals[idx], points[i, j])

    # compute the gradient of the distances with respect to the control points
    gradient = np.zeros(P.shape)
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            for k in range(P.shape[2]):
                # compute the partial derivative of the distance with respect to the control point
                partial = 0.0
                for idx in range(num_points):
                    partial += (current_distances[idx] - distances[idx]) * surface.gradient_point(u_vals[idx], v_vals[idx], i, j, k)

                gradient[i, j, k] = partial

    # update the control points using the gradient
    P -= learning_rate * gradient

    # update the NURBS surface with the new control points
    surface.set_control_points(P)


surface.plot(control_points=True, isocurves_u=8, isocurves_v=8)
plt.show()