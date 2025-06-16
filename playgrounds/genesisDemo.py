# 1. Import and Initialize Genesis
import genesis as gs  # Assuming 'gs' is the common alias

# Initialize Genesis, potentially specifying backend (e.g., CUDA for GPU)
# and other options like logging level.
gs.init(backend=gs.cuda, logging_level="info")  # Or gs.cpu

# 2. Create a Scene
# You can specify simulation options (like timestep, gravity)
# and viewer options (camera position, lookat point, field of view).
sim_opts = gs.options.SimOptions(dt=0.01, gravity=(0, 0, -9.81))
viewer_opts = gs.options.ViewerOptions(
    camera_pos=(3.0, 2.0, 2.0),
    camera_lookat=(0, 0, 0.5),
    # Optional: Adjust camera if needed to better see the interaction
)

scene = gs.Scene(
    sim_options=sim_opts,
    viewer_options=viewer_opts,
    show_viewer=True,  # Set to False for headless mode
)

# 3. Add Entities to the Scene

# Add a ground plane
plane_morph = gs.morphs.Plane()  # Made the plane grey
plane_material = gs.materials.Rigid()
ground_plane = scene.add_entity(
    morph=plane_morph,
    material=plane_material,
)

# Add a red box
# Assuming box size is 0.5x0.5x0.5 for positioning its center
box_size = (0.5, 0.5, 0.5)
red_box_pos = (0.5, 0, box_size[2] / 2.0)  # Place it on the ground
red_box_morph = gs.morphs.Box(
    pos=red_box_pos,
    size=box_size,
)
red_box_material = gs.materials.Rigid()  # Give it some density
red_box = scene.add_entity(
    morph=red_box_morph,
    surface=gs.surfaces.Default(color=(1, 0.5, 0.5)),
    material=red_box_material,
)

# Add a blue cylinder
cylinder_radius = 0.2
cylinder_height = 0.6
# Position it to the left of the box, centered, on the ground
blue_cylinder_pos = (-0.5, 0, cylinder_height / 2.0)
blue_cylinder_morph = gs.morphs.Cylinder(
    pos=blue_cylinder_pos,
    radius=cylinder_radius,
    height=cylinder_height,
    # By default, cylinders are often oriented along Z. If it's along Y or X by default in Genesis,
    # you might need to add: euler=(90, 0, 0) or similar to orient it to roll/push effectively.
    # For now, let's assume it's upright and will push with its side or base.
)
# Make the cylinder a bit denser so it can push effectively
blue_cylinder_material = gs.materials.Rigid()
blue_cylinder = scene.add_entity(
    morph=blue_cylinder_morph,
    surface=gs.surfaces.Default(color=(0, 0, 1)),
    material=blue_cylinder_material,
)

# 4. Build the Scene
# This step finalizes the scene setup and compiles kernels.
scene.build()

# 5. Set Initial Velocity for the Cylinder to Push the Box
# We'll give the blue cylinder a velocity in the positive X direction.
# The exact method might vary, common ones are set_linear_velocity or similar.
# This needs to be called AFTER scene.build() for dynamic objects.
# Assuming the entity object has a method like this:
try:
    # Target velocity (m/s)
    push_velocity = [1.5, 0.0, 0.0]  # Push along positive X-axis
    blue_cylinder.set_linear_velocity(push_velocity)
    print(f"Set initial velocity for blue_cylinder to {push_velocity}")
except AttributeError:
    print(
        "Warning: `blue_cylinder.set_linear_velocity()` method not found directly."
    )
    print(
        "Velocity control might require a different approach in this Genesis version."
    )
    print(
        "The cylinder might remain stationary or only move due to gravity/initial instability."
    )
except Exception as e:
    print(f"Error setting velocity: {e}")


# 6. Run the Simulation Loop
num_simulation_steps = 1000  # Increased steps to see more interaction
for i in range(num_simulation_steps):
    # Advance the simulation by one timestep
    scene.step()

    # (Optional) Get data from the simulation
    if i % 50 == 0:  # Print occasionally
        red_box_current_pos = red_box.get_dofs_position([0])
        blue_cylinder_current_pos = blue_cylinder.get_dofs_position([0])
        print(
            f"Step {i}: Red Box Pos: {red_box_current_pos}, Blue Cylinder Pos: {blue_cylinder_current_pos}"
        )

    # The viewer (if show_viewer=True) should update automatically
    # You might need to add a small sleep if the simulation runs too fast for the viewer
    # import time
    # time.sleep(0.01) # Sleep for 10ms

# 7. (Optional) Clean up or close
# scene.close_viewer() # If applicable
# gs.shutdown()        # If there's a specific shutdown call

print("Simulation finished.")
