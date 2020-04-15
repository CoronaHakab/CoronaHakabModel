# git clone
cd ~
sudo apt-get install -y git
git clone https://github.com/CoronaHakab/CoronaHakabModel.git
cd ./CoronaHakabModel/

# install Python3.8.2
./linux/deb_install_python3.8.sh

# install the project dependencies
sudo pip3.8 install --upgrade pip
sudo pip3.8 install pipenv
pipenv install
pipenv shell

# compile parasymbolic_matrix
sudo apt-get install -y swig
cd ~/CoronaHakabModel/src/corona_hakab_model/parasymbolic_matrix/
python3.8 build_unix.py

#cd ~/CoronaHakabModel/
#echo -e "export QT_QPA_PLATFORM=offscreen\nexport PYTHONPATH=`pwd`/src:`pwd`/src/corona_hakab_model\ncd ~/CoronaHakabModel/\npipenv shell" > ~/.bash_login
#source ~/.bash_login

cd ~/CoronaHakabModel/src/corona_hakab_model
python3.8 ./main.py --help
