from collections import namedtuple
import pathlib
from typing import List

MovingParameter = namedtuple("MovingParameter", "param_name start_range end_range step_size", )


class MovingParametersGenerator:
    @classmethod
    def generate_moving_parameters(cls, params_path, moving_params: List[MovingParameter], output_folder_path):
        """
        Generate parameters files according to the moving parameters and output them to folder.
        """

        with open(params_path, "rt") as read_file:
            loaded_params_str = read_file.read()

        for moving_param in moving_params:
            for param_value in range(moving_param.start_range, moving_param.end_range, moving_param.step_size):
                generated_params = cls.replace_parameter_value(loaded_params_str, moving_param.param_name, param_value)

                cls.write_generated_parameters_to_file(generated_params, moving_param.param_name, param_value,
                                                       output_folder_path)

    @classmethod
    def replace_parameter_value(cls, params_str, param_name, param_value):
        first_str = f"\"{param_name}\": "
        second_str = ","

        first_value_pos = params_str.find(first_str)
        second_value_pos = params_str.find(second_str, first_value_pos)
        return params_str[:first_value_pos + len(first_str)] + str(param_value) + params_str[second_value_pos:]

    @classmethod
    def write_generated_parameters_to_file(cls, generated_params, moving_param_name, moving_param_value,
                                           output_folder_path):
        write_folder_path = f"{output_folder_path}/{moving_param_name}"

        pathlib.Path(write_folder_path).mkdir(parents=True, exist_ok=True)
        with open(f"{write_folder_path}/{moving_param_value}.py", "wt") as write_file:
            write_file.write(generated_params)


if __name__ == "__main__":
    input_file_path = "Parameters/circles_parameters_example.py"
    output_folder_path = "../../output/generated_parameters/circles_parameters"

    population_size_param = MovingParameter(param_name="population_size",
                                            start_range=20_000,
                                            end_range=50_000,
                                            step_size=10_000)
    geo_circles_amount_param = MovingParameter(param_name="geo_circles_amount",
                                               start_range=2,
                                               end_range=5,
                                               step_size=1)
    input_moving_params = [population_size_param, geo_circles_amount_param]

    MovingParametersGenerator.generate_moving_parameters(input_file_path, input_moving_params, output_folder_path)
