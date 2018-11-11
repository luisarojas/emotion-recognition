FILE=./src/run.py
DATA=/tmp/data/
DOCKER_IMAGE=keras-opencv:latest

	# volume path
	# current work
	 # setting home directory ("WORKDIR")
	# mount web camera device
	# ability to display, from host to container
	# file needed for display
	# name of image to create container from
	# run the main python script
run:
	@docker run -it --rm \
	-v $(DATA):/data \
	-v $(PWD):/home/work \
	-w /home/work \
	--device=/dev/video0 \
	-e DISPLAY=$(DISPLAY) \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	$(DOCKER_IMAGE) \
	python $(FILE)

build:
	docker build -t $(DOCKER_IMAGE) \
	-f ./res/docker/Docker-base \
	./res/docker/

jupyter:
	@docker run -it --rm \
	-v $(DATA):/data \
	-v $(PWD):/home/work \
	-w /home/work $(DOCKER_IMAGE)

debug:
	@docker run -it --rm \
	-v $(DATA):/data \
	-v $(PWD):/home/work \
	-w /home/work \
	$(DOCKER_IMAGE)\
	/bin/bash

test:
	docker run --rm -it \
	-v $(PWD):/home/work \
	-w /home/work \
	-e DISPLAY=$(DISPLAY) \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	valian/docker-python-opencv-ffmpeg \
	python ./tutorials/alex-webcam.py
