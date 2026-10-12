# syntax=docker/dockerfile:1
#
# Strata (github.com/Niko1221/Strata) for AMD RDNA3 GPUs through ROCm/HIP.
# Upstream only publishes an NVIDIA (CUDA) image; this one builds the HIP
# engine the way Strata's own setup.py does on a Linux AMD box, so the
# container needs no compiler or ROCm install at runtime.
#
# Build:
#   docker build -t strata-rocm .
#   docker build -t strata-rocm --build-arg STRATA_REF=v0.1.43 .
#
# The engine is compiled for one TheRock GPU family (the ROCm wheels hold one
# family's libraries): gfx110X = RX 7900 / 7800 / 7700 / 7600.

FROM ubuntu:24.04

# Upstream tag, branch or commit; CI passes the resolved release tag
ARG STRATA_REF=main
# Card architectures compiled into the engine; all from the same TheRock family
ARG HIP_ARCHS="gfx1100 gfx1101 gfx1102"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8

# build-essential stays: setup.py compiles the optional CPU image encoder
# (prepare --vision cpu) inside the container on first use
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl git libatomic1 libgomp1 pciutils \
        python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/strata

# fetch by ref so a tag, a branch and a commit sha all work
RUN git init -q . \
    && git remote add origin https://github.com/Niko1221/Strata.git \
    && git fetch -q --depth 1 origin "${STRATA_REF}" \
    && git checkout -q FETCH_HEAD \
    && git log -1 --format='Strata %h %cs %s'

RUN python3 -m venv .venv \
    && .venv/bin/pip install --no-cache-dir --disable-pip-version-check -r requirements.txt

# Reuse setup.py's own HIP build: it installs the pinned TheRock ROCm wheels
# into .venv, compiles with its cmake flags and writes engine/BUILD.json, so a
# later `setup.py` run in the container finds the engine already built.
# STRATA_PORTABLE pins ggml's CPU code to AVX2/FMA/F16C: the default
# -march=native would bake in whatever the build machine has (AVX-512 on many
# CI runners) and die with SIGILL on an AVX2-only CPU. Strata's own AVX-512
# kernels are compiled per file and chosen at runtime either way.
RUN HIP_ARCHS="${HIP_ARCHS}" .venv/bin/python - <<'EOF'
import os
import setup

archs = os.environ["HIP_ARCHS"].split()
setup.isa_floor_defs = lambda floor, bdir, meta: ["-DSTRATA_PORTABLE=ON"]
setup.build_engine_hip({"arch": archs[0], "archs": archs}, setup.get_llama_cpp())
EOF

RUN rm -rf build-hip third_party/*.zip /root/.cache \
    && test -x engine/strata

COPY entrypoint.sh /usr/local/bin/strata-rocm

VOLUME /data
ENTRYPOINT ["strata-rocm"]
CMD ["help"]
