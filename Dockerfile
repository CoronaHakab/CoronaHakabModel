FROM python:3.8.2-buster

ENV WORKDIR=/app
ENV PROJDIR=${WORKDIR}/proj

# git clone
WORKDIR ${WORKDIR}
RUN apt-get update
RUN apt-get -y upgrade
RUN apt-get install -y git
RUN git clone https://github.com/CoronaHakab/CoronaHakabModel.git ${PROJDIR}

WORKDIR ${PROJDIR}
RUN git checkout docker

# install the project dependencies
RUN pip3.8 install --upgrade pip
RUN pip3.8 install pipenv
RUN pipenv install

# compile parasymbolic_matrix
RUN apt-get install -y swig

# we want pipenv run to recognize Pipfile in a few levels up... (the default is 2-3)
ENV PIPENV_MAX_DEPTH=5

# RUN pipenv run python ${PROJDIR}/src/corona_hakab_model/parasymbolic_matrix/build_unix.py
RUN cd ${PROJDIR}/src/corona_hakab_model/parasymbolic_matrix/ \
    && pipenv run python build_unix.py

RUN mkdir ../../output/

CMD git pull \
    && pipenv run python ./src/corona_hakab_model/main.py --help \
    && pipenv run python ./src/corona_hakab_model/main.py generate \
    && pipenv run python ./src/corona_hakab_model/main.py simulate