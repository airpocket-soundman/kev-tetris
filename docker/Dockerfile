# Kev (+ flash-linear-attention, causal-conv1d) on Linux, for the kev-tetris RL loop. See docs/speedup-plan.md.
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS=8
RUN apt-get update && apt-get install -y --no-install-recommends python3.12 python3.12-venv python3.12-dev git ca-certificates build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*
RUN python3.12 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

RUN pip install --upgrade pip wheel setuptools packaging \
    && pip install "torch==2.8.0" --index-url https://download.pytorch.org/whl/cu128

# Kev itself, at the commit deployed on Windows (D:\GitHub\kev, clean at this commit); torch above satisfies its pin
ARG KEV_COMMIT=9098a8b9567968db2dc4ff1add990b66fbb0b100
RUN git clone https://github.com/jaredpalmer/kev.git /opt/kev && git -C /opt/kev checkout "$KEV_COMMIT"     && pip install -e "/opt/kev[serve]"

# the fast paths transformers looks for on Qwen3.5 (DeltaNet and its short convolution)
RUN pip install flash-linear-attention
RUN pip install --no-build-isolation causal-conv1d

WORKDIR /work
