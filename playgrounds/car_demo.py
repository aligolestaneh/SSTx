#!/usr/bin/env python3
"""
Kinodynamic Car Planning in OMPL: RRT vs SST

This demo sets up a simple SE(2) car with forward speed and steering-angle controls,
plans once with RRT and once with SST, and plots both trajectories together.
"""

import numpy as np
from math import sin, cos, tan, sqrt
from functools import partial

import matplotlib.pyplot as plt

try:
    from ompl import base as ob
    from ompl import control as oc
except ImportError:
    # assume OMPL-Python is in ../py-bindings
    import sys
    from os.path import abspath, dirname, join

    sys.path.insert(
        0, join(dirname(dirname(abspath(__file__))), "py-bindings")
    )
    from ompl import base as ob
    from ompl import control as oc


def kinematicCarODE(q, u, qdot):
    """
    q = [x, y, theta]
    u = [v, phi]
    """
    theta = q[2]
    carLength = 0.2
    v, phi = u[0], u[1]
    qdot[0] = v * cos(theta)
    qdot[1] = v * sin(theta)
    qdot[2] = v * tan(phi) / carLength


def isStateValid(si, state):
    # Just check bounds (no obstacles)
    return si.satisfiesBounds(state)


def make_setup():
    # --- State space: SE(2) with x,y ∈ [−2,2], yaw ∈ [0,2π) ---
    space = ob.SE2StateSpace()
    bounds = ob.RealVectorBounds(2)
    bounds.setLow(-2)
    bounds.setHigh(2)
    space.setBounds(bounds)

    # --- Control space: [v,phi] with v∈[0,1], phi∈[-0.5,0.5] ---
    cspace = oc.RealVectorControlSpace(space, 2)
    cb = ob.RealVectorBounds(2)
    cb.setLow(0, 0)
    cb.setHigh(0, 1)
    cb.setLow(1, -0.5)
    cb.setHigh(1, 0.5)
    cspace.setBounds(cb)

    # --- SimpleSetup + validity checker ---
    ss = oc.SimpleSetup(cspace)
    validity = ob.StateValidityCheckerFn(
        partial(isStateValid, ss.getSpaceInformation())
    )
    ss.setStateValidityChecker(validity)

    # --- ODE solver & propagator ---
    ode = oc.ODE(kinematicCarODE)
    odeSolver = oc.ODEBasicSolver(ss.getSpaceInformation(), ode)
    propagator = oc.ODESolver.getStatePropagator(odeSolver)
    ss.setStatePropagator(propagator)

    # --- Start & goal (distance ≈1.414 > 1) ---
    start = ob.State(space)
    start().setX(0.0)
    start().setY(0.0)
    start().setYaw(0.0)
    goal = ob.State(space)
    goal().setX(1.0)
    goal().setY(1.0)
    goal().setYaw(0.0)
    ss.setStartAndGoalStates(start, goal, 0.05)

    return ss


def plan_and_extract(planner_cls, name):
    ss = make_setup()
    si = ss.getSpaceInformation()
    ss.setPlanner(planner_cls(si))
    solved = ss.solve(5.0)
    if not solved:
        print(f"[{name}] planning failed")
        return []
    # Save the states and controls to a file
    with open(f"{name}_states.txt", "w") as f:
        for i in range(ss.getSolutionPath().getStateCount()):
            state = ss.getSolutionPath().getState(i)
            f.write(f"{state.getX()} {state.getY()} {state.getYaw()}\n")
    with open(f"{name}_controls.txt", "w") as f:
        for i in range(ss.getSolutionPath().getControlCount()):
            control = ss.getSolutionPath().getControl(i)
            f.write(f"{control[0]} {control[1]}\n")

    # convert to a geometric path and extract (x,y) samples
    geo = ss.getSolutionPath()
    print(f"Number of states in the path for {name}: {geo.getStateCount()}")
    return [
        (geo.getState(i).getX(), geo.getState(i).getY())
        for i in range(geo.getStateCount())
    ]


if __name__ == "__main__":
    pts_rrt = plan_and_extract(oc.RRT, "RRT")
    pts_sst = plan_and_extract(oc.SST, "SST")

    plt.figure(figsize=(6, 6))
    if pts_rrt:
        xr, yr = zip(*pts_rrt)
        plt.plot(xr, yr, label="RRT", lw=2)
    if pts_sst:
        xs, ys = zip(*pts_sst)
        plt.plot(xs, ys, label="SST", lw=2)

    plt.scatter([0], [0], c="green", marker="o", label="Start")
    plt.scatter([1], [1], c="red", marker="x", label="Goal")
    plt.axis("equal")
    plt.legend()
    plt.title("Kinodynamic Car: RRT vs SST")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()
