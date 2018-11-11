FROM valian/docker-python-opencv-ffmpeg

RUN apt update
RUN apt install -y libsm6 libxext6 libxrender1 libfontconfig1
RUN pip install --upgrade pip
COPY requirements.txt /
RUN pip install -r requirements.txt
