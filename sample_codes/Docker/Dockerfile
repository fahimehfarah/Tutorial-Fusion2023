FROM jupyter/base-notebook
#ENV DEBIAN_FRONTEND noninteractive
MAINTAINER "Luca Zelioli"

# Let's change to root user
USER root

RUN apt-get update && apt-get install -y \
	libopencv-dev \
        python3-pip \
	python3-opencv


# let s change to normal user
USER $NB_UID

# tf and all the others  \
RUN conda install -y -c  numpy pandas matplotlib jupyter pillow
RUN pip install tensorflow
RUN pip install opencv-python
RUN pip install scikit-learn

RUN ["mkdir", "notebooks"]
COPY notebooks/. /home/jovyan/notebooks
#COPY .jupyter /root/.jupyter
#COPY start_jupyter.sh /

# export the labs
ENV JUPYTER_ENABLE_LAB=yes

EXPOSE 8888 6006
