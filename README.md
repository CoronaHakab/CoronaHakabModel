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
- Double click src/corona_hakab_model folder/main.py and run it (Run -> Run)

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
    
    
## Installing and running under Linux (tested on Debian 10):
- sudo apt install swig liblzma-dev libbz2-dev
    - swig - building parasymoblic matrix needs swig
    - liblzma-dev libbz2-dev - since we use Pandas, we need Python with support for bz2 and lzma support
        - (https://stackoverflow.com/questions/22346269/bz2-is-module-not-available-when-installing-pandas-with-pip-in-python-virtual) 
- Building Python 3.8 (currently not on the official repos...)
    - https://linuxize.com/post/how-to-install-python-3-8-on-debian-10/


## Optional - Export/Import matrices!
- Export: **python main.py -o <PATH>**
- Import: **python main.py -i <PATH>**

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
    - List of "moving" parameters - a "moving" parameter contains
        - Parameter name (has to be the same name as in the json file)
        - Start range
        - End range
        - Step size
    - Output folder path - folder path for the json output files
        - Each "moving" parameter will have a separate folder
        - The name of the file is set to be the parameter's value
- Running the module (for now)
    - Manually insert the parameters in code (look at the bottom of the file for example)
    - Go to src/corona_hakab_model folder
    - Run: python moving_parameters_generator.py

## New to git/github?
See the **"How to set up a git environment"** guide in the docs folder.

For Pull-Requests, look at this guide, from step 7 -
[https://product.hubspot.com/blog/git-and-github-tutorial-for-beginners](https://product.hubspot.com/blog/git-and-github-tutorial-for-beginners).

