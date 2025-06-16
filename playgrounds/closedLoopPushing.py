import numpy as np
import genesis as gs
from ompl import base as ob
from ompl import control as oc
from ompl import geometric as og

# Create a simple 2D pushing environment in genesis
gs.init(backend=gs.gpu)

scene = gs.Scene(
    show_viewer=True,
    show_FPS=False,
    sim_options=gs.options.SimOptions(substeps=8),
)

cam = scene.add_camera(
    pos=(0.0, -0.5, 1.0),
    lookat=(0.0, 0.0, 0.0),
    fov=40,
    GUI=False,
)

start_pos = np.array([0.0, -0.2, 0.1])
goal_pos = np.array([0.0, 1.0, 0.1])

plane = scene.add_entity(gs.morphs.Plane())
plane.set_friction(0.2)
cube = scene.add_entity(
    gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(0.0, 0.0, 0.1)),
    surface=gs.surfaces.Default(color=(1.0, 0.0, 0.0)),
    material=gs.materials.Rigid(friction=0.2),
)
rod = scene.add_entity(
    gs.morphs.Cylinder(radius=0.05, height=0.25, pos=start_pos)
)
rod.set_friction(0.2)
scene.build()

# Incremental list from 0 to 0.3 in 100 steps
plan = np.linspace(start_pos[1], goal_pos[1], 500)

for waypoint in plan:
    # make the pos a 1d tensor
    waypoint = np.array([0.0, waypoint, 0.125])
    rod.set_pos(waypoint)
    scene.step()

for i in range(200):
    scene.step()
