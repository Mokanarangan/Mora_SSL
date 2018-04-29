FROM python:3.6

WORKDIR /src/

COPY ./requirements.txt ./requirements.txt
RUN apt-get -qq update && apt-get install --no-install-recommends -y apt-transport-https software-properties-common \
    libzmq1 libzmq-dev python-dev libc-dev pandoc python-pip git-core python-pygraphviz curl \
    build-essential libblas-dev liblapack-dev gfortran \
    libfreetype6-dev libpng-dev net-tools procps python-openbabel pkg-config \
    r-base libreadline-dev

RUN pip install --no-cache-dir -r requirements.txt

