# syntax=docker/dockerfile:1
ARG PG_MAJOR=18

FROM postgres:${PG_MAJOR} AS builder
ARG PG_MAJOR
ARG PGRX_VERSION=0.19.1
ARG FELATAB_MODEL_URL=https://d1ruypri5fhwvl.cloudfront.net/felatab/v1/felatab_int8.safetensors
ARG FELATAB_MODEL_SHA256=547c451a182f4a61aa4ab811efd1fe2e8c57b75b61e4256b28114934dc539741

ENV DEBIAN_FRONTEND=noninteractive \
    RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:/usr/local/rustup/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates build-essential pkg-config \
        libclang-dev clang \
        "postgresql-server-dev-${PG_MAJOR}" \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --profile minimal --default-toolchain stable \
    && rustup component add rustfmt

RUN cargo install cargo-pgrx --version "${PGRX_VERSION}" --locked

RUN cargo pgrx init "--pg${PG_MAJOR}" "$(command -v pg_config)"

WORKDIR /build
COPY Cargo.toml Cargo.lock pg_fela.control build.rs ./
COPY src ./src
COPY tests ./tests

ENV FELATAB_MODEL_URL=${FELATAB_MODEL_URL} \
    FELATAB_MODEL_SHA256=${FELATAB_MODEL_SHA256}
RUN cargo pgrx install --release --pg-config "$(command -v pg_config)"

FROM postgres:${PG_MAJOR}
ARG PG_MAJOR

COPY --from=builder /usr/lib/postgresql/${PG_MAJOR}/lib/pg_fela.so \
                     /usr/lib/postgresql/${PG_MAJOR}/lib/pg_fela.so
COPY --from=builder /usr/share/postgresql/${PG_MAJOR}/extension/pg_fela* \
                     /usr/share/postgresql/${PG_MAJOR}/extension/

COPY docker/initdb/00-create-extension.sql /docker-entrypoint-initdb.d/00-create-extension.sql
