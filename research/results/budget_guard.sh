#!/usr/bin/env bash
# Stops the running E3-bighole app if this program's Modal spend today reaches the cap.
export SSL_CERT_FILE=/root/.ccr/ca-bundle.crt
APP=${1:?usage: budget_guard.sh <app-id> [cap_usd]}; CAP=${2:-8.0}; M=${MODAL:-modal}
cd /tmp
while true; do
  spent=$(timeout 120 $M billing report --for today --resolution h --json 2>/dev/null | python3 -c "
import sys,json
try: d=json.load(sys.stdin)
except Exception: print(-1); sys.exit()
print(round(sum(float(r['cost']) for r in d if r['description'] in ('tpc-experiments','ssh-ring-smoke')),3))")
  state=$(timeout 120 $M app list 2>/dev/null | grep -A0 "$APP" | awk '{print $3}')
  echo "$(date -u +%H:%M) spent=\$$spent app_state=$state"
  if python3 -c "import sys; sys.exit(0 if float('$spent')>=$CAP else 1)"; then
    echo "CAP REACHED: stopping $APP"; timeout 120 $M app stop --yes $APP; exit 0
  fi
  case "$state" in ephemera*|running*|deployed*) ;; *) echo "app no longer running; guard exits"; exit 0;; esac
  sleep 300
done
