import numpy as np

from ompl import control as oc


class PushingControlSampler(oc.ControlSampler):
    def __init__(self, space, obj_shape):
        super().__init__(space)
        self.control_space = space
        self.obj_shape = obj_shape

    def sample(self, control):
        control[0] = np.random.uniform(
            self.control_space.getBounds().low[0],
            self.control_space.getBounds().high[0],
        )
        control[1] = np.random.uniform(
            self.control_space.getBounds().low[1],
            self.control_space.getBounds().high[1],
        )
        control[2] = np.random.uniform(
            self.control_space.getBounds().low[2],
            self.control_space.getBounds().high[2],
        )

        # Convert to absolute control values
        control[0] = int(control[0]) * np.pi / 2
        mask = (control[0] % np.pi) == 0
        control[1] *= np.where(mask, self.obj_shape[1], self.obj_shape[0])
