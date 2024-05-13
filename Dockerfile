FROM mcr.microsoft.com/ccf/app/dev:4.0.7-virtual

# Install Node.js
RUN curl -sL https://deb.nodesource.com/setup_14.x | bash -
RUN apt-get install -y nodejs
RUN apt-get update && \
    apt-get install -y cmake g++ && \
    rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libboost-all-dev



RUN apt-get update && \
    apt-get install -y wget && \
    wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.bz2 && \
    tar xjf eigen-3.4.0.tar.bz2 && \
    rm eigen-3.4.0.tar.bz2

# Create a build directory
# RUN cd eigen-3.4.0
# RUN mkdir build
# RUN cd build
# RUN cmake ..
# RUN mkdir build
# WORKDIR /usr/src/app/build
# RUN cmake ../eigen-3.4.0