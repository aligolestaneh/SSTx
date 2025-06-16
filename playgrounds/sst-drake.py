#!/usr/bin/env python3
"""
Read SST planner output (states & controls) and optimize the trajectory using Drake
via DirectCollocation to shorten path length, then plot both original SST and optimized trajectories.
"""
import numpy as np
import matplotlib.pyplot as plt

# Drake imports
from pydrake.all import (
    VectorSystem,
    DirectCollocation,
    Solve,
    PiecewisePolynomial
)

class KinematicCar(VectorSystem):
    def __init__(self):
        # 2 inputs (v, phi), 0 outputs (we only use continuous state)
        super().__init__(2, 0)
        # continuous state: x, y, theta
        self.DeclareContinuousState(3)
(self):
        # 3 continuous states (x, y, theta), 2 inputs (v, phi)
        super().__init__(3, 2)

    def DoCalcVectorTimeDerivatives(self, context, derivatives):
        x = context.get_continuous_state_vector().CopyToVector()
        u = self.get_input_port(0).Eval(context)
        theta = x[2]
        v, phi = u
        L = 0.2
        # qdot = [vx, vy, vtheta]
        qdot = [v * np.cos(theta), v * np.sin(theta), v * np.tan(phi) / L]
        derivatives.SetFromVector(qdot)

# --- Load SST path ---
sst_states = np.loadtxt("SST_states.txt")   # shape (N, 3)
sst_controls = np.loadtxt("SST_controls.txt")  # shape (N-1, 2)
N = sst_states.shape[0]

# Time grid for initial guess
times = np.linspace(0, 1.0, N)
# Build initial guess splines
x_mat = sst_states.T  # (3, N)
u_mat = np.hstack([sst_controls.T, np.zeros((2,1))])  # pad to N
X_guess = PiecewisePolynomial.FirstOrderHold(times, x_mat)
U_guess = PiecewisePolynomial.FirstOrderHold(times, u_mat)

# Instantiate and convert system
disc_car = KinematicCar().ToAutoDiffXd()
context = disc_car.CreateDefaultContext()

# Setup DirectCollocation
dc = DirectCollocation(
    system=disc_car,
    context=context,
    num_time_samples=N,
    minimum_time_step=0.01,
    maximum_time_step=1.0
)

# State bounds: x,y in [-2,2], theta free; input bounds v∈[0,1], phi∈[-0.5,0.5]
dc.AddBoundingBoxConstraint([
    -2, -2, -np.inf
], [
     2,  2,  np.inf
], dc.state())
dc.AddBoundingBoxConstraint([
    0.0, -0.5
], [
    1.0,  0.5
], dc.input())

# Fix initial/final states
dc.AddLinearConstraint(dc.initial_state() == sst_states[0])
dc.AddLinearConstraint(dc.final_state()   == sst_states[-1])

# Path-length cost on (x,y)
for i in range(N-1):
    xi   = dc.state(i)[:2]
    xnext = dc.state(i+1)[:2]
    dc.AddRunningCost((xnext - xi).dot(xnext - xi))

# Seed with SST solution
dc.SetInitialTrajectory(X_guess, U_guess)

# Solve optimization
aresult = Solve(dc.prog())
if not aresult.is_success():
    raise RuntimeError("Drake optimization failed")
# Reconstruct optimized trajectory
traj = dc.ReconstructTrajectory(aresult)

# Sample optimized trajectory for plotting
ts_plot = np.linspace(0, traj.end_time(), 200)
xy_opt = np.array([traj.value(t)[:2] for t in ts_plot])

# Plot original vs optimized
plt.figure(figsize=(6,6))
plt.plot(sst_states[:,0], sst_states[:,1], '--', label='Original SST', lw=2)
plt.plot(xy_opt[:,0], xy_opt[:,1],      label='Optimized SST', lw=2)
plt.scatter(sst_states[0,0],  sst_states[0,1],  c='green', marker='o', label='Start')
plt.scatter(sst_states[-1,0], sst_states[-1,1], c='red',   marker='x', label='Goal')
plt.title('SST Path: Original vs Drake-Optimized')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.show()
