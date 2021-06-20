FROM python:3.8

WORKDIR /project

RUN python3 -m pip install -U "pip>=21"
COPY requirements.* ./
RUN cat requirements.* > .requirements.txt
RUN python3 -m pip install  -r .requirements.txt


COPY src .
COPY setup.py .
RUN python3 -m pip install --no-deps .

RUN make tests
