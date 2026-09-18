FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

WORKDIR /app

# CBC (the LP solver behind PuLP) ships as a bundled binary and needs no apt
# packages on slim. Install Python deps first so they stay cached.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json ./

# No secrets are baked into this image. API keys are supplied at run time via
# the GEMINI_API_KEYS environment variable.
RUN useradd --create-home --uid 10001 gridwise
USER gridwise

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=4).status==200 else 1)"

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
