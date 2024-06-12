import numpy as np
from .controller import Controller
from models.manipulator_model import ManiuplatorModel


class MMAController(Controller):
    def __init__(self, Tp):
        # TODO: Fill the list self.models with 3 models of 2DOF manipulators with different m3 and r3
        # I:   m3=0.1,  r3=0.05
        # II:  m3=0.01, r3=0.01
        # III: m3=1.0,  r3=0.3
        self.Tp = Tp
        self.u = np.zeros((2,1))
        self.models = [ManiuplatorModel(Tp, m3 = 0.1, r3 = 0.05), ManiuplatorModel(Tp, m3 = 0.01, r3 = 0.01), ManiuplatorModel(Tp, m3 = 1.0, r3 = 0.3)]
        self.i = 0

        self.prev_x = np.zeros(4)
        self.prev_u = np.zeros(2)

    def choose_model(self, x):
        # TODO: Implement procedure of choosing the best fitting model from self.models (by setting self.i)
        x_mi = [model.x_dot(self.prev_x, self.prev_u) * self.Tp + self.prev_x.reshape(4,1) for model in self.models]

        # Model selection - smallest error arg min (x - x_mi)
        error_model_1 = np.sum(abs(x.reshape(4,1) - x_mi[0]))
        error_model_2 = np.sum(abs(x.reshape(4,1) - x_mi[1]))
        error_model_3 = np.sum(abs(x.reshape(4,1) - x_mi[2]))
        errors = [error_model_1, error_model_2, error_model_3]
        min_error = min(errors)
        idx = errors.index(min_error)
        self.i = idx


    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        self.choose_model(x)
        q = x[:2]
        q_dot = x[2:]
        K_d = [[25, 0], [0, 25]]
        K_p = [[60, 0], [0, 60]]
        v = q_r_ddot + K_d @ (q_r_dot - q_dot) + K_p @ (q_r - q)
        M = self.models[self.i].M(x)
        C = self.models[self.i].C(x)
        u = M @ v[:, np.newaxis] + C @ q_dot[:, np.newaxis]
        return u
