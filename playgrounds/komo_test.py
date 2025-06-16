import numpy as np
import matplotlib.pyplot as plt


# ---- 1) Simulate initial zig-zag trajectory (unicycle model) ----
def simulate_controls(v_seq, omega_seq, dt):
    N = len(v_seq)
    xs = np.zeros(N + 1)
    ys = np.zeros(N + 1)
    thetas = np.zeros(N + 1)
    for i in range(N):
        xs[i + 1] = xs[i] + v_seq[i] * np.cos(thetas[i]) * dt
        ys[i + 1] = ys[i] + v_seq[i] * np.sin(thetas[i]) * dt
        thetas[i + 1] = thetas[i] + omega_seq[i] * dt
    return xs, ys, thetas


dt = 0.01
T_total = 2.0
N = int(T_total / dt)
v_seq = np.full(N, np.sqrt(2) / T_total)
omega_seq = 4.0 * np.sign(np.sin(4 * np.pi * np.linspace(0, T_total, N)))
x_init, y_init, theta_init = simulate_controls(v_seq, omega_seq, dt)

# ---- 2) Set up KOMO optimization ----
import libry as ry  # requires RAI/libry installed

# Build simple SE2 car frame
C = ry.Config()
C.addFrame(name="car", parent="", args="transXYPhi")

# KOMO: order=2 for smoothing accelerations
komo = ry.KOMO(C, False, 2, 1, N + 1)
komo.setTiming(N + 1, 1, dt, 2)

# Initialize with the simulated zig-zag
# Flattened initial path: [x0,y0,theta0, x1,y1,theta1, ...]
q_init = np.stack([x_init, y_init, theta_init], axis=1).flatten().tolist()
komo.initWithConstant([0.0, 0.0, 0.0])
komo.setPath(q_init)  # assume API supports setting the initial path

# Control-smoothing objective
komo.addControlObjective([], order=2, scale=1e-2)

# Track the initial trajectory (softly)
for t in range(N + 1):
    komo.addObjective(
        [t],
        ry.FS.framePosition,
        ["car"],
        ry.OT.sos,
        scale=[1e1, 1e1],
        target=[x_init[t], y_init[t]],
    )

# Fix start and end exactly
komo.addObjective(
    [0],
    ry.FS.framePosition,
    ["car"],
    ry.OT.eq,
    scale=[1e3, 1e3],
    target=[0, 0],
)
komo.addObjective(
    [N],
    ry.FS.framePosition,
    ["car"],
    ry.OT.eq,
    scale=[1e3, 1e3],
    target=[1, 1],
)

# ---- 3) Solve and extract the optimized path ----
nlp = komo.nlp()
solver = ry.NLP_Solver()
solver.setProblem(nlp)
opt = solver.getOptions()
opt.set_verbose(1)
solver.setOptions(opt)
solver.solve()

path_opt = np.array(komo.getPath_qAll())
x_opt, y_opt = path_opt[:, 0], path_opt[:, 1]

# ---- 4) Plot raw vs optimized ----
plt.figure(figsize=(6, 6))
plt.plot(x_init, y_init, "-", label="Initial Zig-Zag", alpha=0.5)
plt.plot(x_opt, y_opt, "-", label="KOMO Optimized", linewidth=2)
plt.scatter([1], [1], color="red", marker="x", s=100, label="Goal (1,1)")
plt.xlabel("x position")
plt.ylabel("y position")
plt.title("KOMO Optimization of Zig-Zag Trajectory")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()
