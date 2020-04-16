# Corona Hakab Simulator

## Install and run the simulator from PyCharm (recommended):
- Install Python 3.8.x for Win64
    - Download Windows x86-64 executable installer from **https://www.python.org/downloads/windows/**
    - Install it (i.e. to C:\Python38)
- Install GIT
    - Download and install 64-bit Git for Windows Setup from **https://git-scm.com/**
- Install PyCharm (Community edition is good enough)
    - **https://www.jetbrains.com/pycharm/download/#section=windows**
- VCS -> Checkout from version control -> git
    - **https://github.com/CoronaHakab/CoronaHakabModel.git**
    - choose directory as you please
- File -> Settings -> Project Interpreter ->
    - Click on the plus sign on the right
        - add pipenv
    - Right click on the cog on the top right -> add
        - Pipenv environment ->
            - Base Interpreter Python 3.8 (C:\Python38)
            - Check install packages from pipfile
            - Pipenv executable (C:\Python38\Scripts\pipenv.exe)
            - OK
- In order to make PyCharm recognize the project modules:
    - right click src folder -> Mark directory as -> Source Root
    - right click src/corona_hakab_model folder -> Mark directory as -> Source Root
- double click src/corona_hakab_model folder/main.py and run it (Run -> Run)

## Install and run the simulator from cmd (also possible):
- Install Python 3.8
- pip install pipenv
- run **git clone https://github.com/CoronaHakab/CoronaHakabModel.git** from your dev directory (i.e. c:\dev)
- Create a pipenv environment and install the dependencies
    - full dependencies (for developers): **pipenv install --dev**
    - minimal dependencies (for researches): **pipenv install**
- go to src/corona_hakab_model folder
    - run **set_pythonpath.bat** to add entire src directory to python_path, allowing importing of corona_hakab_model_data.
    - run: **python main.py generate** to generate the population and matrix data.
    - run: **python main.py simulate** to run the simulation with the previously generated population and circles.
    - run: **python main.py all** to run both the generation and simulation.
    - run: **python main.py [simulate|generate] --help** for more help about each option.
    
## Installing on Ubuntu VM
- If running on a remote Ubuntu machine (such as created by Microsoft Azure), run the commands listed in vm_install.txt
- When connecting (with the same user) to the machine, it will automatically setup the environment variables and start the pipenv for CoronaHakabModel.
- Note that when connecting to a machine via SSH, the graphs cannot be displayed, only saved for later viewing. Run **main.py --figure-path image_location** to save the image.

## Optional - Export/Import matrices!
- Export: **python main.py -o <PATH>**
- Import: **python main.py -i <PATH>**

##Optional - Initial sick agent constraints:
- Usage: **python main.py --agent-constraints-path <PATH>**
- Format: csv file with the following columns : geographic_circles,age,Work,School,Family,Other
- each row represents an agent, amount of rows must correspond to initial number of sick agents
- to specify an exact value (geocircle name, age, or number of members in social circle), simply write it in the appropriate column and row
- to specify a range, use '~' eg. "10~70" indicates age 10 to 70, including both
- unspecified values will be free.
- If no agents corresponding to the constraints are found, the code will crash

## Workflow -
- When working, **work on a new git branch**.
- run quality test: **tox -e quality**, if it fails you might need to reformat: **tox -e reformat**.
- if you use named expressions := then for now you have to add the comment # flake8: noqa which will exclude the file from flake8 checks because it doens't support namded expressions currently.
- if ** tox -e reformat** fails then we have to check it out...
- When done, **push changes to your branch, and create a pull request**.
- **Always run the simulator after making changes and before merging**, to make sure you didn't break anything.
- Especially if not sure, **try to get at least 1 person's review before merging**.

## Multiple Jsons generator
- In some cases, we would like to run the simulation/generation multiple times, when all parameters are the same, except 1 or 2 parameters that move across a certain range, in certain steps.
- The module is found in src/corona_hakab_model/moving_parameters_generator.py
- The module parameters
    - Input file path - file path of the json containing the parameters
    - "Moving" parameters file path
        - The file should contain a List of "moving" parameters
        - A "moving" parameter contains
            - Parameter name (has to be the same name as in the json file)
            - Start range
            - End range
            - Step size
    - Output folder path - folder path for the json output files
        - Each "moving" parameter will have a separate folder
        - The name of the file is set to be the parameter's value
- Running the module
    - Go to src/corona_hakab_model folder
    - Run: **python moving_parameters_generator.py --help** for parameters syntax.
    - Run: **python moving_parameters_generator.py** with parameters
    - Run example: **python moving_parameters_generator.py --input-params Parameters/circles_parameters_example.py --moving-params Parameters/moving_parameters_example.json --output-folder ../../output/generated_parameters/circles_parameters**

## Analyzers
- This module is a library that allows a researcher to analyze the output of one or multiple simulation
    - The module implements both specific and generic methods to allow the user flexibility
- To run the module first run the simulation few times, you can do that from the file main.py
- For examples of usage of the module run python src/corona_hakab_model/analyzers/example_script.py, the script plots  a few graphs and demonstrates the usage of the functions.
### Population Analyser
- The population analyser reads a population data file (generated with the simulator, usually "population_data.pickle") and outputs a histogram of population ages, and social circles sizes by type.
- to use, run **python generation_analysis\population_analysis.py**
- Most commonly, you will use the **[-d|--directory]** option to specify the directory to read. The directory is expected to contain a file named "population_data.pickle"
- You can also specify the input population data file using **[-p|--population]**, and output files using **--circle** and **--age**.
- As with all runnables, additional help can be found by running **python generation_analysis\population_analysis.py --help**
### Simulation Analyser
- TODO: Wrap simulation analyser in runnable (argparse, __main__, the works), and add documentation.

## New to git/github?
See the **"How to set up a git environment"** guide in the docs folder.

For Pull-Requests, look at this guide, from step 7 -
[https://product.hubspot.com/blog/git-and-github-tutorial-for-beginners](https://product.hubspot.com/blog/git-and-github-tutorial-for-beginners).

