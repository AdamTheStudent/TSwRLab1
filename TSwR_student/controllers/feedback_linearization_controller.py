import numpy as np
from models.manipulator_model import ManiuplatorModel
from .controller import Controller


class FeedbackLinearizationController(Controller):
    def __init__(self, Tp):
        self.model = ManiuplatorModel(Tp)

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        """
        Please implement the feedback linearization using self.model (which you have to implement also),
        robot state x and desired control v.
        """
        q_t = x[:2]
        q_t_dot = x[2:]

        K_d = [[25, 0], [0, 25]]
        K_p = [[60, 0], [0, 60]]

        v = q_r_ddot + K_d @ (q_r_dot - q_t_dot) + K_p @ (q_r - q_t)

        tau = self.model.M(x) @ v[:, np.newaxis] + self.model.C(x) @ q_r_dot[:, np.newaxis]

        return tau
        # return NotImplementedError()
