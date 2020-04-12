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
    - run: **python main.py**

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

