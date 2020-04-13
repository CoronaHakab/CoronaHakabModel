cd ~
# install Python3.8.2
#https://linuxize.com/post/how-to-install-python-3-8-on-debian-10/
sudo apt update && \
sudo apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libsqlite3-dev libreadline-dev libffi-dev curl liblzma-dev libbz2-dev -y && \
curl -O https://www.python.org/ftp/python/3.8.2/Python-3.8.2.tar.xz && \
tar -xf Python-3.8.2.tar.xz && \
cd Python-3.8.2 && \
./configure --enable-optimizations && \
make -j `nproc` && \
sudo make altinstall && \
python3.8 --version

# git clone
cd ~
sudo apt-get install git -y
git clone https://github.com/CoronaHakab/CoronaHakabModel.git
cd ~/CoronaHakabModel/

# we'll work with docker branch
git checkout docker

# install the project dependencies
sudo pip3.8 install --upgrade pip
sudo pip3.8 install pipenv
pipenv install --dev
pipenv shell

# compile parasymbolic_matrix
sudo apt install -y swig
cd ~/CoronaHakabModel/src/corona_hakab_model/parasymbolic_matrix/
python3.8 build_unix.py

#cd ~/CoronaHakabModel/
#echo -e "export QT_QPA_PLATFORM=offscreen\nexport PYTHONPATH=`pwd`/src:`pwd`/src/corona_hakab_model\ncd ~/CoronaHakabModel/\npipenv shell" > ~/.bash_login
#source ~/.bash_login

cd ~/CoronaHakabModel/src/corona_hakab_model
python3.8 ./main.py --help
