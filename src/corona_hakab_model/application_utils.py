import os

from consts import Consts
from generation.circles_consts import CirclesConsts
from generation.matrix_consts import MatrixConsts
from generation.generation_manager import GenerationManger
from typing import List


MATRIX_CONSTS_FILE_NAME = "matrix_consts.json"
CIRCLES_CONSTS_FILE_NAME = "circles_consts.json"


def generate_from_folder(folder_path: str, output_path: str):
    matrix_consts_path = os.path.join(folder_path, MATRIX_CONSTS_FILE_NAME)
    circles_consts_path = os.path.join(folder_path, CIRCLES_CONSTS_FILE_NAME)

    matrix_consts = MatrixConsts.from_file(matrix_consts_path)
    circles_consts = MatrixConsts.from_file(circles_consts_path)

    GenerationManger(circles_consts=circles_consts, matrix_consts=matrix_consts)
    GenerationManger.export(output_path)  # TODO placeholder - update interface when export/import implemented


def check_folder_for_generation_consts_files(folder_path: str):
    files_list = os.listdir(folder_path)
    return MATRIX_CONSTS_FILE_NAME in files_list and CIRCLES_CONSTS_FILE_NAME in files_list


def pack_simulation_files_to_zip(files: List):
    pass