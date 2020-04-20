# install Python3.8.2
#./linux/deb_install_python3.8.sh

WORKDIR=~
PROJDIR=${WORKDIR}/proj
WORK_BRANCH=develop
GUI_ENABLED=1
#USE_PIPENV=1

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
      if [ "$USE_PIPENV" ]
      then
        sed -i.bak '/pyside2/d' Pipfile
      fi
else
      echo GUI Enabled
      if [ -z "$USE_PIPENV" ]
      then
        sudo pip3 install pyside2
      fi
      sudo apt-get -y install python3-pyqt5
fi

# install the project dependencies
sudo pip3.8 install --upgrade pip
if [ "$USE_PIPENV" ]
then
  sudo pip3.8 install pipenv
  pipenv install
else
  sudo pip3.8 install -r requirements.txt
fi

if [ "$USE_PIPENV" ]
then
  # we want pipenv run to recognize Pipfile in a few levels up... (the default is 2-3)
  export PIPENV_MAX_DEPTH=5
fi

# build parasymbolic_matrix
sudo apt-get install -y swig
cd $PROJDIR/src/corona_hakab_model/parasymbolic_matrix/
pipenv run python build_unix.py
cd $PROJDIR

if [ "$USE_PIPENV" ]
  pipenv run python ./src/corona_hakab_model/main.py --help
else
  python3.8 ./src/corona_hakab_model/main.py --help
fi

echo "in order to run the simulation run the following commands:"
echo "============================================================"
echo "cd $PROJDIR"
if [ "$USE_PIPENV" ]
then
  echo "pipenv run python ./src/corona_hakab_model/main.py all"
else
  echo "python3.8 ./src/corona_hakab_model/main.py all"
fi
echo
echo "To enable Interactive output mode:"
echo "sed -i 's/INTERACTIVE_MODE = False/INTERACTIVE_MODE = True/' $PROJDIR/src/corona_hakab_model/project_structure.py"
echo
echo "To disable Interactive output mode:"
echo "sed -i 's/INTERACTIVE_MODE = True/INTERACTIVE_MODE = False/' $PROJDIR/src/corona_hakab_model/project_structure.py"
