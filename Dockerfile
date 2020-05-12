FROM tensorflow/tensorflow:1.14.0-py3
ADD . /
WORKDIR /
RUN pip install --upgrade pip && \
    pip install --no-cache-dir numpy psutil