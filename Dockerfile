FROM python:3.6

WORKDIR /project

RUN python3 -m pip install -U "pip>=20.2"
COPY requirements.* ./
RUN cat requirements.* > .requirements.txt
RUN python3 -m pip install  -r .requirements.txt


COPY src .
COPY setup.py .
RUN python3 -m pip install --no-deps .

RUN make tests
