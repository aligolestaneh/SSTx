from pathlib import Path

from grapeshot.assets.robot_asset import RobotAsset


UR10 = RobotAsset(
    Path(__file__).parent,
    "ur10_robotis_d435.urdf",
    "ur10_robotis_d435.srdf",
    "manipulator",
)
UR10_ROD = RobotAsset(
    Path(__file__).parent,
    "ur10_rod_d435.urdf",
    "ur10_robotis_d435.srdf",
    "manipulator",
)
