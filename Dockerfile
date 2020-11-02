FROM python:3.6

WORKDIR /project

RUN pip3 install -U "pip>=20.2"
COPY requirements.* ./
RUN cat requirements.* > .requirements.txt
RUN  pip3 install --use-feature=2020-resolver -r .requirements.txt


COPY src .
COPY setup.py .
RUN pip install --no-deps .

RUN make tests
