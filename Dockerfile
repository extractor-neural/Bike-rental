FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=nonintercative
RUN apt-get update -y && apt-get install -y python3 \
	libx11-dev \
	tk \
	python3-tk \
	python3-matplotlib \
	curl \
	git \
	vim \
	python3-pip
WORkDIR /workspace
COPY requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt
CMD ["/bin/bash"]
