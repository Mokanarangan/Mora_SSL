FROM python:3.6

WORKDIR /src/

COPY ./requirements.txt ./requirements.txt
RUN set -x \
    && apk update \
    && apk --no-cache add \
        freetype \
        openblas \
        py3-zmq \
        tini \
    && pip3 install --upgrade pip \
    && ln -s /usr/include/locale.h /usr/include/xlocale.h \
    && apk --no-cache add --virtual .builddeps \
        build-base \
        freetype-dev \
        gfortran \
        openblas-dev \
        python3-dev \
    && pip3 install numpy \
    ## scipy
    && pip3 install scipy \
    ## pnadas 
    && apk --no-cache add  \
        py3-dateutil \
        py3-tz \
    && pip3 install pandas \

RUN pip install --no-cache-dir -r requirements.txt

