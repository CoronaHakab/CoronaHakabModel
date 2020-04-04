from generation.circles_generator import CirclesGenerator
from generation.matrix_generator import MatrixGenerator
from generation.circles_consts import CirclesConsts
from generation.matrix_consts import MatrixConsts


class GenerationManger:
    """
    this class is in charge of the entire generation.
    the generation is built from 2 'stand alone' parts: circles generation and matrix generation
    each gets it's own costs file, and can be imported or exported
    generation manager is in charge of calling each of the sub-parts of the generation, and taking thier results.
    generation manager can export the entire generation information as a json.
    """
    __slots__ = (

    )

    def __init__(self):
        circles_generation = CirclesGenerator(generation_consts=CirclesConsts())
        matrix_generation = MatrixGenerator(circles_generation, matrix_consts=MatrixConsts())

    # todo allow export of the entire generation
    def export(self):
        pass

    # todo allow import of circles generation object, and creation of matrices based on it
    def import_circles_and_create_matrices(self):
        pass

gm = GenerationManger()