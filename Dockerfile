FROM tensorflow/tensorflow

RUN apt-get update
RUN apt update && apt install -y libsm6 libxext6 libxrender1 libfontconfig1
RUN pip install keras
RUN pip install opencv-python
