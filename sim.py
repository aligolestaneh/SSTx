import numpy as np
import genesis as gs
from ompl import base as ob
from ompl import control as oc
from ompl import geometric as og

from geometry.pose import Pose
from functools import partial
from planning.push_to_edge import BoxPropagator
from planning.planning_utils import GraspableRegion


# Simulation class with TCP server
class Sim:
    """Genesis Simulation Environment"""

    def __init__(
        self,
        env_name,
        robot_type,
        object_infos,
        n_envs=100,
        dt=0.01,
        substeps=8,
        convexify=True,
        visualize=True,
    ):
        """
        Initialize with the robot, object type,
        and other critical simulation parameters
        """
        self.env_name = env_name
        self.robot_type = robot_type
        self.object_infos = object_infos
        self.n_envs = n_envs
        self.dt = dt
        self.substeps = substeps
        self.convexify = convexify

        self.show_viewer = visualize
        self.scene = None
        self.env = None
        self.robot = None
        self.objs = []
        self.objs_shape = []
        self.objs_init_poses = []
        # A temporary variable to store the sim angles
        # for the problem of friction anisotropy
        self.sim_angles = np.zeros(n_envs)

        # Set up the environment
        self.init_env()

    def init_env(self):
        # Initialize the Genesis on GPU
        gs.init(backend=gs.gpu)
        # Create the Scene
        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                res=(960, 640), max_FPS=60
            ),
            sim_options=gs.options.SimOptions(
                dt=self.dt, substeps=self.substeps
            ),
            rigid_options=gs.options.RigidOptions(
                box_box_detection=True,
            ),
            vis_options=gs.options.VisOptions(shadow=False),
            show_viewer=self.show_viewer,
            show_FPS=False,
        )

        # Define the states
        default_friction = 0.2

        # Add Ground Plane
        ground_plane = self.scene.add_entity(gs.morphs.Plane())
        # Add Lab scene
        if "lab_scene" in self.env_name:
            table_height = 0.71 + 1e-3  # a bit higher than the actual table
            file_path = (
                "assets/" + self.env_name + "/" + self.env_name + ".urdf"
            )
            self.env = self.scene.add_entity(
                gs.morphs.URDF(
                    file=file_path, fixed=True, convexify=self.convexify
                )
            )
            self.env.set_friction(default_friction)
        else:
            raise NotImplementedError

        # Add Robot
        if "ur10" in self.robot_type:
            self.robot_init = np.array([1.57, -1.7, 2, -1.87, -1.57, 3.14])
            self.robot_base_pose = Pose(np.array([0, 0, table_height]))

            self.robot = self.scene.add_entity(
                gs.morphs.URDF(
                    file="assets/ur10/" + self.robot_type + ".urdf",
                    fixed=True,
                    pos=self.robot_base_pose.position,
                    quat=self.robot_base_pose.rotation,
                    links_to_keep=["ee_link"],
                    convexify=self.convexify,
                )
            )
            # set robot joints
            self.robot_dofs_idx = [0, 1, 2, 3, 4, 5]
            # set end effector
            self.ee_link = self.robot.get_link("ee_link")
            if "pri" in self.robot_type:
                self.ee_dofs_idx = [6, 7]
                self.ee_close_val = 0.053
                self.ee_open_val = 0.0
            else:
                self.ee_dofs_idx = [6, 7, 8, 9]
                self.ee_close_val = 1.12
                self.ee_open_val = 0.0
        else:
            print(f"Robot type {self.robot_type} not supported yet")
            raise NotImplementedError

        # Add Objects
        objs_init_pose = []
        for object_info in self.object_infos:
            if type(object_info) == str:
                object_info = (object_info, (0, -0.5, 0))
            object_type = object_info[0]
            x, y, theta = object_info[1]

            file_path = f"assets/{object_type}/{object_type}.urdf"
            obj = self.scene.add_entity(
                gs.morphs.URDF(file=file_path, convexify=self.convexify)
            )
            obj.set_friction(default_friction)
            self.objs.append(obj)

            obj_init_pose = Pose([x, y, table_height], [0, 0, theta])
            objs_init_pose.append(obj_init_pose)

        # Build the Scene for Parallel Simulation
        self.scene.build(n_envs=self.n_envs, env_spacing=(3.0, 3.0))

        # The following needs to be done after the scene gets built
        # Set Kp and Kd for the Robot control
        self.robot.set_dofs_kp(
            np.full(len(self.robot_dofs_idx), 5000), self.robot_dofs_idx
        )
        self.robot.set_dofs_kv(
            np.full(len(self.robot_dofs_idx), 500), self.robot_dofs_idx
        )
        self.robot.set_dofs_kp(
            np.full(len(self.ee_dofs_idx), 1000), self.ee_dofs_idx
        )
        self.robot.set_dofs_kv(
            np.full(len(self.ee_dofs_idx), 100), self.ee_dofs_idx
        )

        # Get object shape and then set object to given pose
        for i, obj in enumerate(self.objs):
            # object shape
            obj_aabb = obj.get_AABB()[0]
            obj_shape = (obj_aabb[1] - obj_aabb[0]).tolist()
            self.objs_shape.append(obj_shape)

            # update the object initial pose to be on the table
            obj_init_pose = Pose(
                objs_init_pose[i].position
                + np.array([0, 0, obj_shape[2] / 2]),
                objs_init_pose[i].rotation,
            )
            obj_init_poses = [obj_init_pose.copy() for _ in range(self.n_envs)]
            self.objs_init_poses.append(obj_init_poses)

        # Setting everything up
        self.reset()

    ########## Simulation functions ##########
    def run_sim(self, duration, func=None):
        """
        Run the simulation for a given duration,
        also run the function after each step if provided
        """
        for _ in range(int(duration // self.dt)):
            if func is not None:
                func()
            self.scene.step()

    def reset(self, stable_duration=0.5):
        """Reset the robot and objects to the initial state"""
        # A temporary solution for sim angles
        # Reset the scene rotation
        self.rotate_scene(to_angles=0)

        # Reset the robot to the initial state
        self.robot.set_dofs_position(
            np.tile(self.robot_init, (self.n_envs, 1)), self.robot_dofs_idx
        )
        self.robot.control_dofs_position(
            np.tile(self.robot_init, (self.n_envs, 1)), self.robot_dofs_idx
        )

        # Reset the objects to the initial state
        for i, obj in enumerate(self.objs):
            positions = np.array(
                [pose.position for pose in self.objs_init_poses[i]]
            )
            rotations = np.array(
                [pose.rotation for pose in self.objs_init_poses[i]]
            )
            obj.set_pos(positions)
            obj.set_quat(rotations)

        # Run the sim for a few more steps to stabilize
        self.run_sim(stable_duration)

    ########## Object-related functions ##########
    def set_obj_init_poses(self, obj_idx, init_pose, env_idx=None, reset=True):
        """Set the initial pose of the object"""
        if env_idx is not None:
            assert type(init_pose) == Pose, "Initial pose should be a Pose"
            self.objs_init_poses[obj_idx][env_idx] = init_pose

        else:
            # If only one initial pose is provided, repeat it for all envs
            if type(init_pose) == Pose:
                init_pose = [init_pose.copy() for _ in range(self.n_envs)]

            # Should be a list of initial poses when env_idx is not provided
            if len(init_pose) <= self.n_envs:
                self.objs_init_poses[obj_idx][: len(init_pose)] = init_pose
            else:
                raise ValueError("Invalid initial pose size")

        # If required, reset the scene
        if reset:
            self.reset()

    ########## Robot-related functions ##########
    def close_gripper(self, envs_idx=None):
        """Close the gripper"""
        if envs_idx is None:
            gripper_val = np.tile(
                self.ee_close_val, (self.n_envs, len(self.ee_dofs_idx))
            )
        else:
            gripper_val = np.tile(self.ee_close_val, (len(self.ee_dofs_idx)))
        # Control the gripper
        self.robot.control_dofs_position(
            gripper_val, self.ee_dofs_idx, envs_idx
        )

    def open_gripper(self, env_idx=None):
        """Open the gripper"""
        if env_idx is None:
            gripper_val = np.tile(
                self.ee_open_val, (self.n_envs, len(self.ee_dofs_idx))
            )
        else:
            gripper_val = np.tile(self.ee_open_val, (len(self.ee_dofs_idx)))
        # Control the gripper
        self.robot.control_dofs_position(
            gripper_val, self.ee_dofs_idx, env_idx
        )

    def execute_waypoints(self, waypoints):
        """Run the push simulation with the given waypoints"""
        n_run_steps, n_trials, n_joint = waypoints.shape
        assert n_joint == len(self.robot_dofs_idx), "Invalid joint dimension"
        assert n_trials <= self.n_envs, (
            "required number of execution should not be larger"
            + "than the number of simulation environment"
        )

        # Get the current object poses
        all_obj_prev_poses = []
        for obj in self.objs:
            obj_pos = obj.get_pos().cpu().numpy()
            obj_quat = obj.get_quat().cpu().numpy()
            obj_prev_poses = [
                Pose(pos, quat) for pos, quat in zip(obj_pos, obj_quat)
            ]
            all_obj_prev_poses.append(obj_prev_poses)

        data_y = np.zeros((len(self.objs), n_trials, 3))
        # TODO save the whole execution data
        # data_trajs = []
        # data_poses = []
        # data_ts = []

        # Reset the Simulation Environment before execution
        self.robot.set_dofs_position(
            np.tile(self.robot_init, (self.n_envs, 1)), self.robot_dofs_idx
        )
        self.robot.control_dofs_position(
            np.tile(self.robot_init, (self.n_envs, 1)), self.robot_dofs_idx
        )

        # Fill the waypoints with the robot_init to match the size
        if n_trials < self.n_envs:
            complete_waypoints = np.tile(
                self.robot_init, (n_run_steps, self.n_envs, 1)
            )
            complete_waypoints[:, :n_trials, :] = waypoints
            waypoints = complete_waypoints

        # Start to execute robot - Init for each robot
        self.robot.set_dofs_position(waypoints[0], self.robot_dofs_idx)
        self.robot.control_dofs_position(waypoints[0], self.robot_dofs_idx)
        self.scene.step()

        # Run the sim with the computed trajectory
        for k in range(n_run_steps):
            self.robot.control_dofs_position(waypoints[k], self.robot_dofs_idx)
            self.scene.step()

        # Run for a few more steps to stabilize
        self.run_sim(2)

        # Collect data - compute relative SE2 pose
        for i, obj in enumerate(self.objs):
            obj_pos = obj.get_pos().cpu().numpy()
            obj_quat = obj.get_quat().cpu().numpy()
            curr_poses = [
                Pose(pos, rot) for pos, rot in zip(obj_pos, obj_quat)
            ]
            obj_poses = self.get_relative_poses(
                curr_poses, all_obj_prev_poses[i]
            )
            data_y[i, :, :] = obj_poses[:n_trials, :]

        return data_y

    def get_relative_poses(self, current_poses, init_poses):
        """Get the relative SE2 poses of the object"""
        # Compute relative pose
        local_poses = [
            init_pose.invert @ curr_pose
            for curr_pose, init_pose in zip(current_poses, init_poses)
        ]
        # Convert to SE2
        local_pos = np.array([pose.position[:2] for pose in local_poses])
        local_rot = np.array([pose.euler[2:3] for pose in local_poses])
        obj_se2_poses = np.concatenate((local_pos, local_rot), axis=1)

        return obj_se2_poses

    ########## Information functions ##########
    def get_sim_info(self):
        """Return critical infomation"""
        # Stack info
        info = (self.n_envs, self.dt, self.robot_base_pose)
        return info

    def get_obj_info(self, obj_idx):
        """Return critical infomation"""
        # Get the robot and obj poses
        rob_pos = self.robot.get_pos().cpu().numpy()
        rob_quat = self.robot.get_quat().cpu().numpy()
        obj_pos = self.objs[obj_idx].get_pos().cpu().numpy()
        obj_quat = self.objs[obj_idx].get_quat().cpu().numpy()

        robot_poses = [Pose(pos, quat) for pos, quat in zip(rob_pos, rob_quat)]
        obj_poses = [Pose(pos, quat) for pos, quat in zip(obj_pos, obj_quat)]

        # Get the obj poses w.r.t. the robot base
        obj_rob_poses = [
            robot_poses[i].invert @ obj_poses[i] for i in range(self.n_envs)
        ]
        obj_rob_pos = np.array([pose.position for pose in obj_rob_poses])
        obj_rob_quat = np.array([pose.rotation for pose in obj_rob_poses])

        # Object shape
        obj_shape = self.objs_shape[obj_idx]

        # Stack info
        info = (
            obj_pos,
            obj_quat,
            obj_rob_pos,
            obj_rob_quat,
            obj_shape,
        )
        return info

    # A temporary solution for the problem of friction anisotropy
    # Rotate the whole scene by a certain yaw angle
    def rotate_scene(self, by_angles=None, to_angles=None):
        """
        Rotate the whole scene
        by a certain yaw angle or to a certain yaw angle
        """
        if by_angles is None and to_angles is None:
            return

        # Determine the new angles for all envs.
        if to_angles is not None:
            if isinstance(to_angles, float) or isinstance(to_angles, int):
                to_angles = np.full(self.n_envs, to_angles)
            self.sim_angles = np.array(to_angles)
        else:
            self.sim_angles = self.sim_angles + np.array(by_angles)

        # New rotation for the scene
        new_rots = np.array(
            [
                Pose(rotation=(0, 0, angle)).rotation
                for angle in self.sim_angles
            ]
        )

        # Store previous table pose
        prev_table_pos = self.env.get_pos().cpu().numpy()
        prev_table_quat = self.env.get_quat().cpu().numpy()
        prev_table_poses = [
            Pose(pos, quat)
            for pos, quat in zip(prev_table_pos, prev_table_quat)
        ]
        # Set the new table pose
        self.env.set_quat(new_rots)
        new_table_poses = [
            Pose(prev_table_pos[i], new_rots[i]) for i in range(self.n_envs)
        ]

        # Set the new robot base pose
        self.robot.set_quat(new_rots)

        # Set the new object pose based on the previous table pose
        # Keep relative pose w.r.t. the table
        for obj in self.objs:
            obj_pos = obj.get_pos().cpu().numpy()
            obj_ori = obj.get_quat().cpu().numpy()
            obj_poses = [Pose(pos, ori) for pos, ori in zip(obj_pos, obj_ori)]

            obj_local_poses = [
                prev_table_poses[i].invert @ obj_poses[i]
                for i in range(self.n_envs)
            ]
            obj_new_poses = [
                new_table_poses[i] @ obj_local_poses[i]
                for i in range(self.n_envs)
            ]
            obj.set_pos(np.array([pose.position for pose in obj_new_poses]))
            obj.set_quat(np.array([pose.rotation for pose in obj_new_poses]))

        # Update the scene
        self.scene.step()

    def plan_to_edge(
        self,
        start,
        goal,
        planner="SST",
        planning_time=10.0,
        obj_idx=0,
        bin=False,
    ):
        """Plan to edge"""
        space = ob.SE2StateSpace()

        bounds = ob.RealVectorBounds(2)
        bounds.setLow(0, -0.9)
        bounds.setHigh(0, 0.70)
        bounds.setLow(1, -1.2)
        bounds.setHigh(1, 0.0)
        space.setBounds(bounds)

        cspace = oc.RealVectorControlSpace(space, 3)

        cbounds = ob.RealVectorBounds(3)
        cbounds.setLow(0, 0)  # minimum rotation
        cbounds.setHigh(0, 4)  # maximum rotation
        cbounds.setLow(1, -0.4)  # minimum side offset
        cbounds.setHigh(1, 0.4)  # maximum side offset
        cbounds.setLow(2, 0.0)  # minimum push distance
        cbounds.setHigh(2, 0.3)  # maximum push distance
        cspace.setBounds(cbounds)

        ss = oc.SimpleSetup(cspace)
        si = ss.getSpaceInformation()

        propagator = BoxPropagator(self.objs_shape[obj_idx])

        def isStateValid(spaceInformation, state):
            if bin:
                # Check the collision of the object with obj_idx
                collision_pairs = self.objs[obj_idx].detect_collision()
                # Delete the collision pairs with the lab_scene
                collision_pairs = [
                    pair
                    for pair in collision_pairs
                    if pair[0] != self.env.idx and pair[1] != self.env.idx
                ]
                valid = len(
                    collision_pairs
                ) == 0 and spaceInformation.satisfiesBounds(state)
                # print(f"[INFO] Valid: {valid}")
                return valid
            else:
                return spaceInformation.satisfiesBounds(state)

        ss.setStateValidityChecker(
            ob.StateValidityCheckerFn(partial(isStateValid, si))
        )
        ss.setStatePropagator(oc.StatePropagatorFn(propagator.propagate))

        start_state = ob.State(space)
        start_state().setX(start[0])
        start_state().setY(start[1])
        start_state().setYaw(start[2])
        ss.setStartState(start_state)

        if bin:
            goal_state = BinRegion(
                ss.getSpaceInformation(),
                goal,
                self.objs_shape[obj_idx],
                -0.76,
            )
        else:
            goal_state = GraspableRegion(
                ss.getSpaceInformation(),
                goal,
                self.objs_shape[obj_idx],
                0.76,
            )
        goal_state.setThreshold(0.01)
        ss.setGoal(goal_state)
        planner = oc.SST(si) if planner == "SST" else oc.RRT(si)
        ss.setPlanner(planner)

        si.setPropagationStepSize(3.0)
        si.setMinMaxControlDuration(1, 1)  # Control duration in steps

        solved = ss.solve(planning_time)

        if solved:
            print(f"[INFO]: Planning to edge successful!")
            path = ss.getSolutionPath()

            states = []
            for i in range(path.getStateCount()):
                state = path.getState(i)
                states.append([state.getX(), state.getY(), state.getYaw()])

            controls = []
            durations = []
            for i in range(path.getControlCount()):
                control = path.getControl(i)
                controls.append(
                    [int(control[0]) * np.pi / 2, control[1], control[2]]
                )
                durations.append(path.getControlDuration(i))

            return states, controls, durations
        else:
            print(f"[ERROR]: Planning to push edge failed!")
            return None, None, None

    def plan_to_grasp(
        self,
        obj_idx,
        planner="RRTConnect",
        planning_time=1.0,
        num_waypoints=100,
    ):
        """Plan to grasp"""
        space = ob.CompoundStateSpace()
        [space.addSubspace(ob.SO2StateSpace(), 1.0) for _ in range(6)]
        ss = og.SimpleSetup(space)

        def is_state_valid(state):
            self.robot.set_qpos(self.state2array(state), self.robot_dofs_idx)
            collision_pairs = self.robot.detect_collision()
            return len(collision_pairs) == 0

        ss.setStateValidityChecker(
            ob.StateValidityCheckerFn(partial(is_state_valid))
        )
        ss.setPlanner(getattr(og, planner)(ss.getSpaceInformation()))
        start_state = ob.State(space)
        for i in range(6):
            start_state[i] = float(
                self.robot.get_dofs_position()[0, i].cpu().numpy()
            )

        ss.setStartState(start_state)
        obj_pos, obj_quat, _, _, obj_shape = self.get_obj_info(obj_idx)
        # goal_state = GraspPointRegion(
        #     ss.getSpaceInformation(),
        #     obj_shape,
        #     obj_pos,
        #     obj_quat,
        #     self.robot,
        # )
        grasps = self.generate_grasp(obj_idx)
        grasp = grasps[np.random.randint(0, len(grasps))]
        print(grasp)
        goal_state = ob.State(space)
        for i in range(6):
            goal_state[i] = float(grasp[0, i].cpu().numpy())
        ss.setGoalState(goal_state)
        solved = ss.solve(planning_time)
        waypoints = []
        if solved:
            print(f"[INFO]: Planning to grasp successful!")
            path = ss.getSolutionPath()
            ps = og.PathSimplifier(ss.getSpaceInformation())
            try:
                ps.partialShortcutPath(path)
                ps.ropeShortcutPath(path)
            except:
                ps.shortcutPath(path)

            ps.smoothBSpline(path)

            if num_waypoints is not None:
                path.interpolate(num_waypoints)
            waypoints = self.states2array(path.getStates())
        else:
            print(f"[ERROR]: Planning to grasp failed!")

        return waypoints

    def is_grasp_valid(self, grasp):
        self.robot.set_qpos(grasp, self.robot_dofs_idx)
        collision_pairs = self.robot.detect_collision()
        return len(collision_pairs) == 0

    def generate_grasp(self, obj_idx):
        """Generate grasp points for a given object."""
        grasps = []
        w, l, h = self.objs_shape[obj_idx]
        for edge in range(4):
            rotation = edge * np.pi / 2

            if edge % 2 == 1:
                offset_size = w
                to_edge = l / 2
            else:
                offset_size = l
                to_edge = w / 2

            for offset in np.linspace(-0.4, 0.4, 10):
                offset *= offset_size
                pre_push_offset = -0.02
                offset_pose = Pose([0, 0, pre_push_offset])

                dir_vector = np.array([np.cos(rotation), np.sin(rotation)])
                side_offset_vector = np.array([-dir_vector[1], dir_vector[0]])

                start = (dir_vector * (to_edge + pre_push_offset)) + (
                    offset * side_offset_vector
                )

                local_pos = [start[0], start[1], 0]

                for i in range(2):
                    match edge:
                        case 0:
                            reflect_z_euler = [0, np.pi / 2, 0]
                            rotate_z_euler = [0, 0, i * np.pi + np.pi / 2]
                        case 1:
                            reflect_z_euler = [np.pi / 2, 0, 0]
                            rotate_z_euler = [0, 0, i * np.pi]
                        case 2:
                            reflect_z_euler = [0, -np.pi / 2, 0]
                            rotate_z_euler = [0, 0, i * np.pi + np.pi / 2]
                        case 3:
                            reflect_z_euler = [-np.pi / 2, 0, 0]
                            rotate_z_euler = [0, 0, i * np.pi]

                    reflect_z = Pose(rotation=reflect_z_euler)
                    rotate_z = Pose(rotation=rotate_z_euler)
                    surface = reflect_z @ rotate_z  # order?
                    local_quat = surface.rotation

                    obj_pos, obj_quat, _, _, _ = self.get_obj_info(obj_idx)
                    obj_init_pose = Pose(obj_pos[0], obj_quat[0])
                    start_pose = obj_init_pose @ Pose(local_pos, local_quat)

                    pos = np.array(
                        [
                            [
                                start_pose.position[0],
                                start_pose.position[1],
                                obj_pos[0][2] + h / 2,
                            ]
                        ]
                    )
                    quat = np.array([start_pose.rotation])
                    ee_link = self.robot.get_link("ee_link")
                    q = self.robot.inverse_kinematics(
                        link=ee_link,
                        pos=pos,
                        quat=quat,
                    )
                    q = q[:, :6]
                    if self.is_grasp_valid(q):
                        grasps.append(q)

        return grasps

    def state2array(self, state):
        array = np.empty(len(self.robot_dofs_idx))
        for i in range(len(self.robot_dofs_idx)):
            array[i] = state[i].value
        return array

    def states2array(self, states):
        array_list = []
        for state in states:
            array_list.append(self.state2array(state))
        return array_list


if __name__ == "__main__":
    # TEST
    sim = Sim(
        "lab_scene_bin",
        "ur10_robotis_d435_pri",
        [("wood_block_flipped", (0.75, -0.7, 0))],
        n_envs=100,
        dt=0.01,
        substeps=3,
        convexify=True,
    )

    input()
    sim.close_gripper([0])
    for _ in range(1000):
        sim.scene.step()
