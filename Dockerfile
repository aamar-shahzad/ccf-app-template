FROM mcr.microsoft.com/ccf/app/dev:4.0.7-virtual

# Install Node.js
RUN curl -sL https://deb.nodesource.com/setup_14.x | bash -
RUN apt-get install -y nodejs
