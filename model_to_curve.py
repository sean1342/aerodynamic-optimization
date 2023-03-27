import nurbspy as nrb
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# define the list of points to fit to
points = np.array([[(0.0, 0.0, 0.0), (1.0, 0.0, 1.0), (2.0, 0.0, 0.0)],
                   [(0.0, 1.0, 1.0), (1.0, 1.0, 0.0), (2.0, 1.0, 1.0)],
                   [(0.0, 2.0, 0.0), (1.0, 2.0, 1.0), (2.0, 2.0, 0.0)]
])

# P is the control points that we modify
# initialize P to the points to fit to to start off
P = points

# transpose to get in proper format
P = np.array(P).transpose((2, 1, 0))

# create the NURBS surface
surface = nrb.NurbsSurface(control_points=P)

# define loss function
# loss = sum(distance from point to surface point) ** 2
def loss_function(control_points):
    loss = 0
    return loss


initial_guess = P.flatten()

result = minimize(loss_function, initial_guess, method='BFGS')
optimized_control_points = np.reshape(result.x, P.shape)

# plot the final surface with the control points
surface.plot(control_points=True, isocurves_u=8, isocurves_v=8)
plt.show()
