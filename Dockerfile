FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1
# Optional: git for HF models that reference git LFS, and basic deps
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# copy app source
COPY . .

# default: no-bill local mode
ENV NM_USE_BQ=0
ENV LOG_SINK=local
ENV PORT=8080
EXPOSE 8080

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8080"]
