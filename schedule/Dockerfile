FROM ubuntu:16.04
RUN apt update -y
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y python-pip python-dev build-essential vim
RUN pip install --upgrade pip
RUN pip install requests
RUN pip install requests --upgrade 
RUN DEBIAN_FRONTEND=noninteractive pip install kubernetes pandas numpy
RUN mkdir data
COPY scheduler.py scheduler.py
WORKDIR /
ENTRYPOINT ["python"]
CMD ["scheduler.py"]
