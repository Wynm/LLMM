#!/usr/bin/env bash
# تجهيز البيئة من الصفر. آمن للتشغيل المتكرّر.
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
echo "▸ بايثون: $($PY -V)"
echo "▸ تثبيت المتطلّبات…"
$PY -m pip install --quiet --upgrade pip 2>/dev/null || true
$PY -m pip install --quiet -r requirements.txt

echo "▸ تحقّق سريع…"
$PY - <<'EOF'
import numpy, fastapi, uvicorn
from mishkat.text import normalize
print("  numpy", numpy.__version__, "| fastapi", fastapi.__version__)
print("  تطبيع:", normalize("المَدْرَسَةُ"))
EOF

echo "▸ اكتمل التجهيز. جرّب:  make test   أو   make run"
