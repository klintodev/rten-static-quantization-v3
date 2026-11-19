FROM rust:1.94.0-bookworm@sha256:365468470075493dc4583f47387001854321c5a8583ea9604b297e67f01c5a4f

WORKDIR /app
ARG RTEN_COMMIT=e77a3b6e12650512449f66843b6508ec913ebd17
RUN git init -q . \
    && git fetch -q --depth=1 https://github.com/robertknight/rten.git "$RTEN_COMMIT" \
    && git checkout -q --detach FETCH_HEAD \
    && test "$(git rev-parse HEAD)" = "$RTEN_COMMIT" \
    && git submodule update --init --recursive

ENV CARGO_TARGET_DIR=/opt/rten-target \
    CARGO_BUILD_JOBS=2 \
    CARGO_INCREMENTAL=0 \
    RUST_TEST_THREADS=2 \
    RAYON_NUM_THREADS=2

# Only the build fetches dependencies. Keep the test compiler cache outside
# /app so Harbor transfers just the working tree to the separate verifier.
RUN cargo fetch --locked \
    && cargo test --locked --offline -p rten --features all-ops,rten_format --lib --no-run \
    && cargo test --locked --offline -p rten-shape-inference --lib --no-run \
    && python3 --version

ENV CARGO_NET_OFFLINE=true
RUN cargo build --locked --offline --release -p rten-cli
CMD ["bash"]
