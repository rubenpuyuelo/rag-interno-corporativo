#!/usr/bin/env bash
set -euo pipefail

API="http://127.0.0.1:8000"

echo "Waiting for API /ready..."
for i in {1..60}; do
  if curl -fsS "$API/ready" | python -c "import sys, json; d=json.load(sys.stdin); sys.exit(0 if d.get('ready') is True else 1)" >/dev/null 2>&1; then
    echo "API is ready."
    break
  fi
  sleep 2
done

echo "Test 1: vacaciones"
curl -fsS -X POST "$API/preguntar" -H "Content-Type: application/json" \
  -d '{"question":"¿Cuántos días de vacaciones tengo?"}' \
  | python -c "import sys, json; d=json.load(sys.stdin); assert '23' in d['answer']; assert len(d['citations'])>=1; print('OK')"

echo "Test 2: fuera de docs"
curl -fsS -X POST "$API/preguntar" -H "Content-Type: application/json" \
  -d '{"question":"¿Cuál es el menú del comedor?"}' \
  | python -c "import sys, json; d=json.load(sys.stdin); assert d['answer']=='NO ENCONTRADO'; assert d['citations']==[]; print('OK')"

echo "Smoke tests passed."
