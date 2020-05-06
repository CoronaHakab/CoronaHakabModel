import logging
import os.path

from generation.circles_consts import CirclesConsts
from generation.circles_generator import CirclesGenerator
from generation.matrix_consts import MatrixConsts
from generation.matrix_generator import MatrixGenerator


class GenerationManger:
    """
    this class is in charge of the entire generation.
    the generation is built from 2 'stand alone' parts: circles generation and matrix generation
    each gets it's own costs file, and can be imported or exported
    generation manager is in charge of calling each of the sub-parts of the generation, and taking thier results.
    generation manager can export the entire generation information as a json.
    """

    __slots__ = ("matrix_data", "population_data", "connection_data", "circles_consts", "matrix_consts")

    def __init__(self, circles_consts: CirclesConsts, matrix_consts: MatrixConsts):
        # setting logger
        logger = logging.getLogger("generation")
        logging.basicConfig()
        logger.setLevel(logging.INFO)

        logger.info("creating population and circles")
        circles_generation = CirclesGenerator(circles_consts=circles_consts)
        self.population_data = circles_generation.population_data

        logger.info("creating connections and matrix")
        matrix_generation = MatrixGenerator(circles_generation.population_data, matrix_consts=matrix_consts)
        self.matrix_data = matrix_generation.matrix_data
        self.connection_data = matrix_generation.connection_data


        # save consts to allow export
        self.circles_consts = circles_consts
        self.matrix_consts = matrix_consts

    def save_to_folder(self, folder):
        self.matrix_data.export(os.path.join(folder, 'matrix_data'))
        self.population_data.export(folder, 'population_data')
        self.connection_data.export(folder, "connection_data")
        self.circles_consts.export(folder, 'circles_consts.json')
        self.matrix_consts.export(folder, 'matrix_consts.json')
