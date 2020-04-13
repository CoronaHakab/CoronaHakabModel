# install Python3.8.2 on Debian 10 or Ubuntu 18.04
#https://linuxize.com/post/how-to-install-python-3-8-on-ubuntu-18-04/
#https://linuxize.com/post/how-to-install-python-3-8-on-debian-10/

sudo apt update -y && \
sudo apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libsqlite3-dev libreadline-dev libffi-dev curl liblzma-dev libbz2-dev && \
curl -O https://www.python.org/ftp/python/3.8.2/Python-3.8.2.tar.xz && \
tar -xf Python-3.8.2.tar.xz && \
cd Python-3.8.2 && \
./configure --enable-optimizations && \
make -j `nproc` && \
sudo make altinstall && \
python3.8 --version
