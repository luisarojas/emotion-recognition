FILE=./src/run.py
DATA=/tmp/data/
DOCKER_IMAGE=keras-opencv:latest

run:
	docker run -it --rm \
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
	./res/docker/