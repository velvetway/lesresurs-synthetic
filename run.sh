#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/backend"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  . .venv/bin/activate
  pip install -q --upgrade pip
  pip install -q -r requirements.txt
else
  . .venv/bin/activate
fi
echo "ЛесРесурс: http://localhost:8000"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
