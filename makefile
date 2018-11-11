DOCKER_IMAGE=keras-opencv:latest

docker-build:
	docker build -t $(DOCKER_IMAGE) .
