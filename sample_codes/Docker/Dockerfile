MAINTAINER "Luca Zelioli"
FROM ubuntu:20.04
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y \
	libopencv-dev \
        python3-pip \
	python3-opencv && \
    rm -rf /var/lib/apt/lists/* \

# tf and all the others  \
RUN pip3 install tensorflow
RUN pip3 install numpy
RUN pip3 install pandas
RUN pip3 install sklearn
RUN pip3 install matplotlib
RUN pip3 install jupyter
RUN pip3 install opencv-python
RUN pip3 install pillow

RUN ["mkdir", "notebooks"]
COPY conf/.jupyter /root/.jupyter
COPY start_jupyter.sh /

EXPOSE 8888 6006
VOLUME /notebooks
CMD ["/start_jupyter.sh"]