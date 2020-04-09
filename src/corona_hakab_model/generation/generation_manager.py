import logging

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

    __slots__ = ("matrix_data", "population_data")

    def __init__(self, circles_consts: CirclesConsts):
        # setting logger
        logger = logging.getLogger("generation")
        logging.basicConfig()
        logger.setLevel(logging.INFO)

        logger.info("creating population and circles")
        circles_generation = CirclesGenerator(circles_consts=circles_consts)
        self.population_data = circles_generation.population_data

        logger.info("creating connections and matrix")
        matrix_generation = MatrixGenerator(circles_generation.population_data, matrix_consts=MatrixConsts())
        self.matrix_data = matrix_generation.matrix_data

    # todo allow export of the entire generation
    def export(self):
        pass

    # todo allow import of circles generation object, and creation of matrices based on it
    def import_circles_and_create_matrices(self):
        pass
