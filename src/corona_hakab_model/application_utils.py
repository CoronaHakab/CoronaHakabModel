import logging
import os

from consts import Consts
from generation.circles_consts import CirclesConsts
from generation.matrix_consts import MatrixConsts
from generation.generation_manager import GenerationManger
from typing import List
from zipfile import ZipFile

MATRIX_CONSTS_FILE_NAME = "matrix_consts.json"
CIRCLES_CONSTS_FILE_NAME = "circles_consts.json"

logger = logging.getLogger("application")


def generate_from_folder(folder_path: str):
    matrix_consts_path = os.path.join(folder_path, MATRIX_CONSTS_FILE_NAME)
    circles_consts_path = os.path.join(folder_path, CIRCLES_CONSTS_FILE_NAME)

    matrix_consts = MatrixConsts.from_file(matrix_consts_path)
    circles_consts = MatrixConsts.from_file(circles_consts_path)

    gm = GenerationManger(circles_consts=circles_consts, matrix_consts=matrix_consts)
    gm.save_to_folder(folder_path)  # TODO placeholder - update interface when export/import implemented
    logger.info(f"Generation output files were saved in {folder_path}")


def generate_from_master_folder(master_folder_path: str):
    """
    Generate simulation data for each viable sub-folder
    """
    viable_subfolders = get_viable_subfolders_for_generation(master_folder_path)
    logger.info(f"Found {len(viable_subfolders)} viable sub-folders.")
    for folder in viable_subfolders:
        # TODO multi-threading
        generate_from_folder(folder)


def get_viable_subfolders_for_generation(master_folder_path: str):
    """
    Find all immediate sub-folders that have both const files present
    """
    # get all sub-directories and ignore files
    sub_folders = [os.path.join(master_folder_path, path)
                   for path
                   in os.listdir(master_folder_path)
                   if os.path.isdir(os.path.join(master_folder_path, path))]
    # Filter only folders containing consts files
    viable_subfolders = [folder for folder in sub_folders if check_folder_for_generation_consts_files(folder)]
    return viable_subfolders


def check_folder_for_generation_consts_files(folder_path: str):
    """
    Check if folder is viable for simulation data generation -
    both matrix and circles consts files should be present.
    """
    files_list = os.listdir(folder_path)
    return MATRIX_CONSTS_FILE_NAME in files_list and CIRCLES_CONSTS_FILE_NAME in files_list
