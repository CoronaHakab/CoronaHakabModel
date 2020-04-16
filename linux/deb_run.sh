# install Python3.8.2
./linux/deb_install_python3.8.sh

export PROJDIR=~/CoronaHakabModel/
export WORKBRANCH=develop

# git clone
cd ~
sudo apt-get update
sudo apt-get -y upgrade
sudo apt-get install -y git
git clone https://github.com/CoronaHakab/CoronaHakabModel.git $PROJDIR

cd $PROJDIR
git checkout $WORKBRANCH

# install the project dependencies
sudo pip3.8 install --upgrade pip
sudo pip3.8 install pipenv
pipenv install

# we want pipenv run to recognize Pipfile in a few levels up... (the default is 2-3)
export PIPENV_MAX_DEPTH=5

# build parasymbolic_matrix
sudo apt-get install -y swig
cd $PROJDIR/src/corona_hakab_model/parasymbolic_matrix/
python3.8 build_unix.py

#mkdir output

git pull
cd $PROJDIR/src/corona_hakab_model
pipenv run python ./main.py --help
pipenv run python main.py generate
pipenv run python main.py simulate