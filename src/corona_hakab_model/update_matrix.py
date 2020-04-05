from typing import Iterable

import numpy as np
from generation.connection_types import ConnectionTypes
from manager import SimulationManager


class UpdateMatrixManager:
    """
    Manages the "Update Matrix" stage of the simulation.
    """

    def __init__(self, manager: "SimulationManager"):
        self.manager = manager
        # unpacking commonly used information from manager
        self.matrix = manager.matrix
        self.matrix_type = manager.matrix_type
        self.depth = manager.depth
        self.logger = manager.logger
        self.consts = manager.consts
        self.size = len(manager.agents)
        # todo unpack more important information
        self.normalize_factor = None
        self.total_contagious_probability = None
        self.normalize()

    def normalize(self):
        """
        this function should normalize the weights within W to represent the infection rate.
        As r0=bd, where b is number of daily infections per person
        """
        self.logger.info(f"normalizing matrix")
        if self.normalize_factor is None:
            # updates r0 to fit the contagious length and ratio.
            states_time = self.consts.average_time_in_each_state()
            total_contagious_probability = 0
            for state, time_in_state in states_time.items():
                total_contagious_probability += time_in_state * state.contagiousness
            beta = self.consts.r0 / total_contagious_probability

            # saves this for the effective r0 graph
            self.total_contagious_probability = total_contagious_probability

            # this factor should be calculated once when the matrix is full, and be left un-changed for the rest of the run.
            self.normalize_factor = (beta * self.size) / (self.matrix.total())

        self.matrix *= self.normalize_factor  # now each entry in W is such that bd=R0

    def change_connections_policy(self, connection_types_to_use: Iterable[ConnectionTypes]):
        self.logger.info(f"changing policy. keeping all matrices of types: {connection_types_to_use}")
        factors = np.zeros(self.depth, dtype=np.float32)
        for connection_type in connection_types_to_use:
            ind = connection_type.value
            factors[ind] = 1
        self.matrix.set_factors(factors)
        self.normalize()

    def update_matrix_step(self):
        """
        Update the matrix step
        """
        # for now, we will not update the matrix at all
        pass
