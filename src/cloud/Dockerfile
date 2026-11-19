# The review job: everything one analysis needs, in one image (built from the context bin/cloud-deploy stages).
FROM python:3.14-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
ENV HF_HOME=/opt/hf PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# CPU-only PyTorch first: the default Linux wheel carries several GB of GPU libraries the job cannot use.
COPY requirements.txt .
RUN pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt

# The sentence encoder at the revision the library's embeddings were made with.
ARG ENCODER_REVISION=e8c3b32edf5434bc2275fc9bab85f82640a19130
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-mpnet-base-v2', revision='$ENCODER_REVISION')" \
    && mkdir -p /opt/hf/hub/models--sentence-transformers--all-mpnet-base-v2/refs \
    && printf '%s' "$ENCODER_REVISION" > /opt/hf/hub/models--sentence-transformers--all-mpnet-base-v2/refs/main
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

COPY . .
# The build fails here if a file the library needs was not staged.
RUN python -c "from src.tools.predict_retention import tool; print(len(tool.state().lib), 'library videos'); tool.encoder()" \
    && chmod -R a+rX /app /opt/hf
USER 65534
ENV HOME=/tmp
ENTRYPOINT ["python", "-m", "src.cloud.review_job"]
