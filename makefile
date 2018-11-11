FILE=./src/run.py
DATA=/tmp/data/
DOCKER_IMAGE=keras-opencv:latest

run:
	@docker run -it --rm \
	-v $(DATA):/data \ # volume path
	-v $(PWD):/home/work \ # current work
	-w /home/work \ # setting home directory ("WORKDIR")
	--device=/dev/video0 \ # mount web camera device
	-e DISPLAY=$(DISPLAY) \ # ability to display, from host to container
	-v /tmp/.X11-unix:/tmp/.X11-unix \ # file needed for display
	$(DOCKER_IMAGE) \ # name of image to create container from
	python $(FILE) # run the main python script

jupyter:
	@docker run -it --rm \
	-v $(DATA):/data \
	-v $(PWD):/home/work \
	-w /home/work $(DOCKER_IMAGE)

test:
	@docker run -it --rm \
	-v $(DATA):/data \
	-v $(PWD):/home/work \
	-w /home/work \
	$(DOCKER_IMAGE)\
	/bin/bash

build:
	docker build -t $(DOCKER_IMAGE) \
	-f ./res/docker/Docker-base \
	./res/docker/
