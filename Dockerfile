FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/root/.cache/huggingface \
    CTEC_WHISPER_MODEL=small \
    CTEC_MAX_TEXT_CHARS=120000 \
    CTEC_MAX_REFERENCE_BYTES=31457280 \
    CTEC_MAX_RESULT_BASE64_BYTES=14680064

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
        git \
        libgomp1 \
        libsndfile1 \
        rubberband-cli \
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
