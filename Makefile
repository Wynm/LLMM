.PHONY: help bootstrap test cov lint build run bench clean stats corpus-check

PY ?= python3
PORT ?= 8000

help:
	@echo "مِشكاة — أوامر التطوير"
	@echo "  make bootstrap    تثبيت المتطلّبات"
	@echo "  make test         تشغيل الاختبارات"
	@echo "  make cov          الاختبارات مع التغطية"
	@echo "  make build        بناء كل الفهارس (فهرس + متجهات + رسم)"
	@echo "  make run          تشغيل الاستوديو على المنفذ $(PORT)"
	@echo "  make bench        تقرير أداء"
	@echo "  make stats        إحصاءات المجموعة"
	@echo "  make clean        حذف المخرجات المُشتقّة"

bootstrap:
	@bash bootstrap.sh

test:
	$(PY) -m pytest -q

cov:
	$(PY) -m pytest -q --cov=mishkat --cov-report=term-missing

build:
	$(PY) -m mishkat.cli build --all

run:
	$(PY) -m mishkat.cli serve --port $(PORT)

bench:
	$(PY) -m mishkat.cli bench

stats:
	$(PY) -m mishkat.cli stats

corpus-check:
	$(PY) -m mishkat.cli corpus-check

clean:
	rm -rf artifacts .pytest_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
