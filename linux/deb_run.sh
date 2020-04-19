# install Python3.8.2
#./linux/deb_install_python3.8.sh

export WORKDIR=~
export PROJDIR=${WORKDIR}/proj
export WORK_BRANCH=paths_fix_3.7.0

# git clone
cd $WORKDIR
sudo apt-get update
sudo apt-get -y upgrade
sudo apt-get install -y git
git clone https://github.com/CoronaHakab/CoronaHakabModel.git $PROJDIR

cd $PROJDIR

git checkout $WORK_BRANCH
sed -i.bak '/pyside2/d' Pipfile

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

pipenv run python $PROJDIR/src/corona_hakab_model/main.py --help
pipenv run python $PROJDIR/src/corona_hakab_model/main.py all
