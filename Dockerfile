# مِشكاة — صورةٌ قابلة للتشغيل بلا شبكة
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# بناء الفهارس داخل الصورة حتى يبدأ الخادم فورًا
RUN python -m mishkat.cli corpus-check && python -m mishkat.cli build

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=4s --start-period=10s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health').status==200 else 1)"

CMD ["python", "-m", "mishkat.cli", "serve", "--port", "8000", "--host", "0.0.0.0"]
