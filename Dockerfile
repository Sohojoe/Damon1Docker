FROM continuumio/miniconda
MAINTAINER Joe Booth "joe@joebooth.com"
RUN apt-get update -y
RUN apt-get install -y python-pip python-dev build-essential
COPY damon1 PYTHONPATH/damon1
WORKDIR /damon_install
ADD ./requirements.txt /damon_install/requirements.txt
RUN pip install -r requirements.txt
