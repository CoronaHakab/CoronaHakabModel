FROM python:3.8.2-buster

ENV WORKDIR=/app
ENV PROJDIR=${WORKDIR}/proj
ENV WORK_BRANCH=develop
ARG GUI_ENABLED=1

# git clone
WORKDIR ${WORKDIR}
RUN apt-get update
RUN apt-get -y upgrade
RUN apt-get install -y git
RUN git clone https://github.com/CoronaHakab/CoronaHakabModel.git ${PROJDIR}

WORKDIR ${PROJDIR}

RUN git checkout -b ${WORK_BRANCH}

RUN if [ -z "$GUI_ENABLED" ] ; \
    then \
        echo GUI Disabled
    else \
        echo GUI Enabled && \
        pip3.8 install PySide2
        apt-get -y install python3-pyqt5 ; \
    fi

# install the project dependencies
RUN pip3.8 install --upgrade pip
RUN pip3.8 install -r requirements.txt

# build parasymbolic_matrix
RUN apt-get install -y swig
RUN cd ${PROJDIR}/src/corona_hakab_model/parasymbolic_matrix/ \
    && python3.8 build_unix.py

CMD python3.8 ./src/corona_hakab_model/main.py --help
