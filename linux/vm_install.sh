sudo apt-get update 
sudo apt-get install checkinstall -y
sudo apt-get install libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev -y
sudo apt-get install liblzma-dev -y
cd /opt
sudo wget https://www.python.org/ftp/python/3.8.2/Python-3.8.2.tgz
sudo tar xzf Python-3.8.2.tgz
cd Python-3.8.2
sudo ./configure --enable-optimizations
sudo -H make altinstall
cd ~
git clone https://github.com/CoronaHakab/CoronaHakabModel.git
cd ~/CoronaHakabModel/
sudo chmod 777 -R /usr/local/lib/python3.8/
sudo python3.8 ./setup.py install
sudo pip3.8 install pipenv
cd ~/CoronaHakabModel/
pipenv install --dev
pipenv shell
sudo apt install -y swig
cd ~/CoronaHakabModel/src/corona_hakab_model/parasymbolic_matrix/
sudo pip3.8 install --upgrade pip
sudo pip3.8 install swimport
python3.8 build_unix.py 
pip install pandas
cd ~/CoronaHakabModel/
echo -e "export QT_QPA_PLATFORM=offscreen\nexport PYTHONPATH=`pwd`/src:`pwd`/src/corona_hakab_model\ncd ~/CoronaHakabModel/\npipenv shell" > ~/.bash_login
source ~/.bash_login
cd ~/CoronaHakabModel/src/corona_hakab_model
python3.8 ./main.py --help
