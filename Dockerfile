# French Lab helper on Render (free web service). Same contents as the Modal image in modal_app.py.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# Piper: free, open-source text-to-speech with a French voice baked into the image (no credits, no account)
RUN pip install --no-cache-dir -r requirements.txt "piper-tts>=1.3"
RUN mkdir -p /voices && cd /voices \
    && curl -fsSL -O https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/siwis/medium/fr_FR-siwis-medium.onnx \
    && curl -fsSL -O https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/siwis/medium/fr_FR-siwis-medium.onnx.json

COPY app.py .
ENV PYTHONUNBUFFERED=1 FL_DATA_DIR=/tmp/fl-data
# Render sets PORT; app.py serves on it
CMD ["python", "app.py"]
