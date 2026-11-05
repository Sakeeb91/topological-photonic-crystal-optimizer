#!/usr/bin/env bash
# Prepare a Modal client inside a Claude Code cloud session (UNTESTED end to end: api.modal.com
# was blocked when this was written). Not needed on your own laptop.
#
# Why: this container's HTTPS goes through an agent proxy that re-terminates TLS with its own CA
# (/root/.ccr/ca-bundle.crt). The Modal client talks gRPC via grpclib, which trusts only the
# certifi bundle, and it needs the optional python-socks extra to tunnel through HTTPS_PROXY.
# This script installs modal[api-proxy-support] in a venv and appends the proxy CA to that
# venv's certifi bundle (adds trust; verification stays on).
#
# Requires: api.modal.com and api.modal2.com in the environment's Allowed domains, and
# MODAL_TOKEN_ID / MODAL_TOKEN_SECRET set as environment variables (see docs/MODAL_RUNBOOK.md).
set -euo pipefail
V=${V:-$HOME/.venv-modal}
python3 -m venv "$V"
"$V/bin/pip" install -q "modal[api-proxy-support]" pyyaml numpy
if [ -f /root/.ccr/ca-bundle.crt ]; then
  CERT=$("$V/bin/python" -c "import certifi; print(certifi.where())")
  grep -q "ccr-agent-proxy-ca" "$CERT" || { echo "# ccr-agent-proxy-ca" >> "$CERT"; cat /root/.ccr/ca-bundle.crt >> "$CERT"; }
fi
[ -n "${MODAL_TOKEN_ID:-}" ] && [ -n "${MODAL_TOKEN_SECRET:-}" ] && echo "Modal token env vars: present" || echo "Modal token env vars: MISSING"
"$V/bin/modal" profile current 2>/dev/null || true
echo "Use: $V/bin/modal run research/modal/app.py::e0"
