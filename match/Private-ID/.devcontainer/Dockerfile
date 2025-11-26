FROM rust:latest AS build

ENV BASE /usr/local
RUN apt-get update && \
    apt-get install -y cmake
RUN rustup install stable && \
rustup component add rustfmt --toolchain stable-x86_64-unknown-linux-gnu
