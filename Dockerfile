FROM rocm/tensorflow:latest

RUN apt update
RUN apt dist-upgrade -y
RUN apt install -y libnuma-dev

RUN wget -qO - http://repo.radeon.com/rocm/rocm.gpg.key | apt-key add -
RUN echo deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main | tee /etc/apt/sources.list.d/rocm.list
RUN apt-get install -y rocm-dkms
#RUN update-initramfs -u
RUN adduser root video

RUN apt install rocm-libs rccl
RUN pip3 install --user tensorflow-rocm --upgrade
RUN pip3 install --user matplotlib

RUN mkdir -p /root/data
COPY ./address_parser address_parser/
COPY ./saved_models saved_models/
COPY train.py setup.py predict.py ./
RUN python3 setup.py build
RUN python3 setup.py install --user

ENTRYPOINT python3 ./train.py
