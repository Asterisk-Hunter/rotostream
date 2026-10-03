FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ROTOSTREAM_WORKSPACE_DIR=/workspace \
    ROTOSTREAM_DEVICE=cpu

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 rotostream \
    && useradd --uid 10001 --gid rotostream --create-home rotostream \
    && mkdir /workspace \
    && chown rotostream:rotostream /workspace

WORKDIR /app
COPY api/requirements-runtime.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=rotostream:rotostream api/app ./app

USER rotostream
EXPOSE 8010
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8010/api/ready', timeout=3)"
# In-process jobs and GPU gates require one worker. Never scale workers on a shared workspace.
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8010", "--workers", "1", "--no-server-header", "--timeout-graceful-shutdown", "60"]

