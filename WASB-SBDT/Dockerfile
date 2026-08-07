FROM nvcr.io/nvidia/tensorrt:21.06-py3

LABEL maintainer="shuhei.tarashima@ntt.com"
LABEL OBJECT="2D ball tracking (WASB/HRNet fork) for the project's 2D tracking step"
ARG http_proxy
ARG https_proxy
ENV http_proxy ${http_proxy}
ENV https_proxy ${https_proxy}
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends\
	apt-utils\
	apt-transport-https\
	ca-certificates\
	software-properties-common\
&& apt-get clean\
&& rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y \
	git\
	vim\
	cmake\
	libsm6\ 
	libxext6\
	libxrender-dev\
	libgl1-mesa-glx\
	locales\
	build-essential\
	zlib1g-dev\
	libssl-dev\
	libbz2-dev\
	libreadline-dev\
	libsqlite3-dev\
	libhdf5-dev\
	libncursesw5-dev\
	libgdbm-dev\
	liblzma-dev\
	libdb-dev\
	libffi-dev\
	uuid-dev\
	tk-dev\
&& apt-get clean\
&& rm -rf /var/lib/apt/lists/*

ENV LANG en_US.UTF-8  
ENV LANGUAGE en_US:en  
ENV LC_ALL en_US.UTF-8
RUN locale-gen en_US.UTF-8

RUN pip3 install numpy==1.22.4 torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
# Trimmed down from the original repo's deps (hydra-core, tqdm, scikit-learn,
# scikit-image, pandas, einops, timm, matplotlib): this fork bypasses the
# original hydra-based training/eval framework entirely (see eval_wasb.py,
# track_ball.py), so none of those are needed. attrdict/PyYAML added
# permanently here since load_model() needs them (previously reinstalled by
# hand each session, see claude_context.md).
RUN pip3 install opencv-python==4.6.0.66 attrdict PyYAML

COPY src /root/src
WORKDIR /root/src
#COPY .vimrc /root/

