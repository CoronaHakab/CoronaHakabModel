# install Python3.8.2
#./linux/deb_install_python3.8.sh

export WORKDIR=~
export PROJDIR=${WORKDIR}/proj
export WORK_BRANCH=develop_linux
export GUI_ENABLED=1

# git clone
cd $WORKDIR
sudo apt-get update
sudo apt-get -y upgrade
sudo apt-get install -y git
git clone https://github.com/CoronaHakab/CoronaHakabModel.git $PROJDIR

cd $PROJDIR

git checkout $WORK_BRANCH
if [ -z "$GUI_ENABLED" ]
then
      echo GUI Disabled
      sed -i.bak '/pyside2/d' Pipfile
else
      echo GUI Enabled
      sudo apt-get -y install python3-pyqt5
fi

# install the project dependencies
sudo pip3.8 install --upgrade pip
sudo pip3.8 install pipenv
pipenv install

# we want pipenv run to recognize Pipfile in a few levels up... (the default is 2-3)
export PIPENV_MAX_DEPTH=5

# build parasymbolic_matrix
sudo apt-get install -y swig
cd $PROJDIR/src/corona_hakab_model/parasymbolic_matrix/
pipenv run python build_unix.py
cd $PROJDIR

pipenv run python ./src/corona_hakab_model/main.py --help
#pipenv run python ./src/corona_hakab_model/main.py all

echo "in order to run the simulation run the following commands:"
echo "============================================================"
echo "cd $PROJDIR"
echo "pipenv run python ./src/corona_hakab_model/main.py all"
echo
echo "To enable Interactive output mode:"
echo "sed -i 's/INTERACTIVE_MODE = False/INTERACTIVE_MODE = True/' $PROJDIR/src/corona_hakab_model/project_structure.py"
echo
echo "To disable Interactive output mode:"
echo "sed -i 's/INTERACTIVE_MODE = True/INTERACTIVE_MODE = False/' $PROJDIR/src/corona_hakab_model/project_structure.py"
