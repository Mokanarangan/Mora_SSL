FROM python:3.6

WORKDIR /src/

COPY ./requirements.txt ./requirements.txt
RUN apt-get install --no-install-recommends -y build-essential libblas-dev 

RUN pip install --no-cache-dir -r requirements.txt

