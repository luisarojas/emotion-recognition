FILE=src/run-emotion.py
DOCKER_IMAGE=keras-opencv:latest
DATA_PATH=/home/alex/Downloads/data

run:
	@docker run -it --rm -v $(DATA_PATH):/data -v $(PWD):/home/work -w /home/work $(DOCKER_IMAGE) python $(FILE)

jupyter:
	@docker run -it --rm -v $(DATA_PATH):/data -v $(PWD):/home/work -w /home/work $(DOCKER_IMAGE)

test:
	@docker run -it --rm -v $(DATA_PATH):/data -v $(PWD):/home/work -w /home/work $(DOCKER_IMAGE) /bin/bash

docker-build:
	docker build -t $(DOCKER_IMAGE) ./res/docker/
