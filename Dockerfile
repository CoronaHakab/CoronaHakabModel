FROM python:3.8.2-buster

ENV WORKDIR=/app
ENV PROJDIR=${WORKDIR}/proj

# change this to develop once docker branch is merged this needs to be deleted
ENV WORKBRANCH=develop

# git clone
WORKDIR ${WORKDIR}
RUN apt-get update
RUN apt-get -y upgrade
RUN apt-get install -y git
RUN git clone https://github.com/CoronaHakab/CoronaHakabModel.git ${PROJDIR}

WORKDIR ${PROJDIR}
RUN git checkout ${WORKBRANCH}

# install the project dependencies
RUN pip3.8 install --upgrade pip
RUN pip3.8 install pipenv
RUN pipenv install

# we want pipenv run to recognize Pipfile in a few levels up... (the default is 2-3)
ENV PIPENV_MAX_DEPTH=5

# build parasymbolic_matrix
RUN apt-get install -y swig
RUN cd ${PROJDIR}/src/corona_hakab_model/parasymbolic_matrix/ \
    && pipenv run python build_unix.py

#RUN mkdir output

CMD git pull \
    && cd ./src/corona_hakab_model/ \
    && pipenv run python main.py --help \
    && pipenv run python main.py generate \
    && pipenv run python main.py simulate