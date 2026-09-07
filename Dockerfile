FROM ghcr.io/ggml-org/whisper.cpp:main-vulkan

ARG ENV_TYPE=production
RUN if [ "$(echo $ENV_TYPE | tr A-Z a-z)" = "development" ]; then \
    apt update && \
    apt install -y --no-install-recommends python3-pip python3-venv && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*; \
  fi

COPY pyproject.toml /tmp/digue/pyproject.toml
RUN if [ "$(echo $ENV_TYPE | tr A-Z a-z)" = "development" ]; then \
    python3 -m venv /opt/venv && \
    /opt/venv/bin/python -m pip install --upgrade "pip>=25.1" && \
    /opt/venv/bin/python -m pip install --group /tmp/digue/pyproject.toml:dev && \
    rm -rf /tmp/digue; \
  fi

ENV PATH="/opt/venv/bin:$PATH"
