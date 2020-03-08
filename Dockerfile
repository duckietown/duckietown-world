FROM python:3.6

WORKDIR /project

COPY requirements.* ./

RUN pip install -r requirements.txt

COPY src .
COPY setup.py .
RUN pip install --no-deps .

RUN make tests
