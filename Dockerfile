FROM quay.io/centos/centos:stream9

RUN dnf -y update && \
    dnf -y install epel-release && \
    dnf -y install \
    gcc \
    gcc-c++ \
    make \
    cmake \
    pkg-config \
    wget \
    tar \
    python3 \
    python3-pip \
    openssl-devel \
    gettext \
    ca-certificates && \
    dnf clean all

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install Envoy
ENV ENVOY_VERSION=1.31.7
RUN curl -OL https://github.com/envoyproxy/envoy/releases/download/${ENVOY_VERSION}/envoy-${ENVOY_VERSION}-linux-x86_64 
RUN chmod +x envoy-${ENVOY_VERSION}-linux-x86_64
RUN mv envoy-${ENVOY_VERSION}-linux-x86_64 /usr/local/bin/envoy

# Install Golang
ENV GOLANG_VERSION=1.24.1
RUN curl -OL https://golang.org/dl/go${GOLANG_VERSION}.linux-amd64.tar.gz && \
    tar -C /usr/local -xzf go${GOLANG_VERSION}.linux-amd64.tar.gz && \
    rm go${GOLANG_VERSION}.linux-amd64.tar.gz
ENV PATH="/usr/local/go/bin:${PATH}"
ENV GOPATH="/go"
ENV PATH="/go/bin:${PATH}"

# Set working directory
WORKDIR /app

# Set environment variables
ENV LD_LIBRARY_PATH=/app/candle-binding/target/release
ENV CGO_ENABLED=1
