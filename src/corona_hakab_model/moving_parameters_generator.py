from argparse import ArgumentParser
from dataclasses import dataclass
import numpy as np
import pathlib
import json
import os


@dataclass
class MovingParameter:
    parameter_name: str
    start_range: float
    end_range: float
    step_size: float


class MovingParametersGenerator:
    @classmethod
    def generate_moving_parameters(cls, params_path, moving_params_path, output_folder_path):
        """
        Generate parameters files according to the moving parameters and output them to folder.
        """

        with open(params_path, "rt") as params_file:
            loaded_params_str = params_file.read()

        moving_params = cls.get_moving_parameters(moving_params_path)

        for moving_param in moving_params:
            param_values = np.arange(moving_param.start_range, moving_param.end_range, moving_param.step_size)
            for param_value in param_values:
                generated_params = cls.replace_parameter_value(loaded_params_str, moving_param.parameter_name,
                                                               param_value)

                cls.write_generated_parameters_to_file(generated_params, moving_param.parameter_name, param_value,
                                                       output_folder_path)

    @classmethod
    def get_moving_parameters(cls, moving_params_path):
        """
        Load moving parameters from file and return a list of MovingParameter.
        """

        with open(moving_params_path, "rt") as moving_params_file:
            moving_params = json.load(moving_params_file)

        return [MovingParameter(**moving_param_dict) for moving_param_dict in moving_params]

    @classmethod
    def replace_parameter_value(cls, params_str, param_name, param_value):
        """
        Replace value of parameter name in parameters string and return replaced string.
        """

        first_str = f"\"{param_name}\": "
        second_str = ","

        first_str_index = params_str.find(first_str)
        second_str_index = params_str.find(second_str, first_str_index)
        return params_str[:first_str_index + len(first_str)] + str(param_value) + params_str[second_str_index:]

    @classmethod
    def write_generated_parameters_to_file(cls, generated_params, moving_param_name, moving_param_value,
                                           output_folder_path):
        """
        Write value of generated_params to output file.
        File path is output folder path + moving parameter name + moving parameter value.
        Create directories of file path if not exists.
        """

        write_folder_path = os.path.join(output_folder_path, moving_param_name)

        pathlib.Path(write_folder_path).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(write_folder_path, f"{moving_param_value}.py"), "wt") as write_file:
            write_file.write(generated_params)


if __name__ == "__main__":
    parser = ArgumentParser("Multiple Jsons Generator")
    parser.add_argument("-p",
                        "--input-params",
                        dest="input_file_path",
                        help="Input parameters file path")
    parser.add_argument("-m",
                        "--moving-params",
                        dest="moving_parameters_file_path",
                        help="Moving parameters file path")
    parser.add_argument("-o",
                        "--output-folder",
                        dest="output_folder_path",
                        default='../../output/generated_parameters',
                        help="Output folder - default is  ../../output/generated_parameters")
    args = parser.parse_args()

    MovingParametersGenerator.generate_moving_parameters(args.input_file_path, args.moving_parameters_file_path,
                                                         args.output_folder_path)
