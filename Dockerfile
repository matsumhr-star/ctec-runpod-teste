FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/root/.cache/huggingface \
    CTEC_QWEN_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    CTEC_MAX_TEXT_CHARS=120000 \
    CTEC_MAX_REFERENCE_BYTES=31457280 \
    CTEC_MAX_RESULT_BASE64_BYTES=14680064 \
    CTEC_QWEN_CHUNK_CHARS=650
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
        git \
        libgomp1 \
        libsndfile1 \
        sox \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN python -m pip install --upgrade \
        "pip==25.1.1" \
        "setuptools==75.8.0" \
        "wheel==0.45.1" \
    && python -m pip install -r requirements.txt
COPY handler.py ./handler.py
CMD ["python", "-u", "handler.py"]
