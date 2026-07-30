# syntax=docker/dockerfile:1.7

FROM node:22-bookworm-slim@sha256:f32b81066cde10a75dbac96646099533316d94bac4150c55da1636e1f0ffdc46 AS web-build

WORKDIR /build/web/checkers
COPY web/checkers/package.json web/checkers/package-lock.json ./
RUN npm ci
COPY web/checkers/ ./
RUN npm run build

FROM python:3.12-slim-bookworm@sha256:d50fb7611f86d04a3b0471b46d7557818d88983fc3136726336b2a4c657aa30b AS runtime

ENV MALLOC_ARENA_MAX=2 \
    OMP_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY deploy/checkers/requirements.lock /app/requirements.lock
RUN python -m pip install --no-cache-dir --require-hashes --requirement /app/requirements.lock \
    && rm /app/requirements.lock \
    && useradd --system --uid 10001 --create-home --home-dir /home/checkers checkers

COPY src/checkers/ /app/src/checkers/
COPY scripts/serve_checkers_web.py /app/scripts/serve_checkers_web.py
COPY --from=web-build /build/web/checkers/dist/ /app/static/

USER 10001:10001
EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/api/health', timeout=2).read()"]

ENTRYPOINT ["python", "/app/scripts/serve_checkers_web.py"]
CMD ["--bundle", "/app/model/policy.pt", "--static-dir", "/app/static", "--port", "8765", "--max-active-games", "256", "--idle-ttl-seconds", "21600", "--max-concurrent-requests", "8", "--request-timeout-seconds", "15"]
