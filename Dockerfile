FROM docker:18-dind

WORKDIR /duckietown-shell

COPY requirements.txt .

RUN apk --update --no-cache add \
	python2 \
	python2-dev \
	py-pip \
	bash \
	git \
	gcc \
	musl-dev \
	linux-headers \
    && pip install -r requirements.txt \
    && apk del python2-dev gcc musl-dev linux-headers

# copy the rest  
COPY . .

#   Note the contents of .dockerignore:
#
#     **
#     !requirements.txt
#     !lib
#     !setup.py
#     !README.md
#
#   That's all we need - do not risk including spurious files.


# Install the package using '--no-deps': you want to pin everything
# using requirements.txt
# So, we want this to fail if we forgot anything.
#RUN pip install --prefix /usr --no-deps .

COPY . .

RUN pip install .

ENV PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

#ENTRYPOINT ["dts"]
