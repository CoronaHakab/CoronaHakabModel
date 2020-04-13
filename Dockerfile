FROM ubuntu:18.04

ENV WORKDIR=/app

WORKDIR ${WORKDIR}

# install Python3.8.2
#https://linuxize.com/post/how-to-install-python-3-8-on-debian-10/
RUN apt-get update -y \
    && apt-get install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libsqlite3-dev libreadline-dev libffi-dev curl liblzma-dev libbz2-dev
RUN curl -O https://www.python.org/ftp/python/3.8.2/Python-3.8.2.tar.xz \
    && tar -xf Python-3.8.2.tar.xz
RUN cd Python-3.8.2 \
    && ./configure --enable-optimizations \
    && make -j `nproc` \
    && make altinstall \
    && python3.8 --version \
    && cd ..

ENV PROJDIR=${WORKDIR}/proj

# git clone
WORKDIR ${WORKDIR}
RUN apt-get install -y git
RUN git clone https://github.com/CoronaHakab/CoronaHakabModel.git ${PROJDIR}

WORKDIR ${PROJDIR}
RUN git checkout docker

# install the project dependencies
RUN pip3.8 install --upgrade pip
RUN pip3.8 install pipenv
RUN pipenv install --dev

# compile parasymbolic_matrix
RUN apt-get install -y swig

# we want pipenv run to recognize Pipfile in a few levels up... (the default is 2-3)
ENV PIPENV_MAX_DEPTH=5

# RUN pipenv run python ${PROJDIR}/src/corona_hakab_model/parasymbolic_matrix/build_unix.py
RUN cd ${PROJDIR}/src/corona_hakab_model/parasymbolic_matrix/ \
    && pipenv run python build_unix.py

#cd ~/CoronaHakabModel/
#echo -e "export QT_QPA_PLATFORM=offscreen\nexport PYTHONPATH=`pwd`/src:`pwd`/src/corona_hakab_model\ncd ~/CoronaHakabModel/\npipenv shell" > ~/.bash_login
#source ~/.bash_login

# RUN apt-get install qt5-default

CMD ["pipenv", "run", "python", "./src/corona_hakab_model/main.py", "--help"]