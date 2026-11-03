# Base image with PyTorch + CUDA pre-installed
FROM nvcr.io/nvidia/pytorch:25.10-py3

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# uv for fast pip installs at build and runtime
COPY --from=ghcr.io/astral-sh/uv:0.8.18 /uv /usr/local/bin/uv
ENV UV_BREAK_SYSTEM_PACKAGES=1

# Copy the source allowed by .dockerignore for the editable install.
# Rebuild after source changes, or explicitly bind-mount a checkout at /app.
COPY . /app/

# Snapshot the base image's installed packages and use them as a constraints
# file for our install. The NVIDIA base ships a custom CUDA torch
# (2.9.0a0+...) and a torchvision built against it; without these pins
# uv resolves to PyPI's vanilla torch and breaks the torchvision ABI
# (operator torchvision::nms missing) at runtime.
RUN python -m pip freeze --exclude-editable \
        | grep -ivE "^(morph|-e )" \
        > /tmp/base_constraints.txt \
    && head -50 /tmp/base_constraints.txt

RUN uv pip install --system -c /tmp/base_constraints.txt -e .

CMD ["python"]
