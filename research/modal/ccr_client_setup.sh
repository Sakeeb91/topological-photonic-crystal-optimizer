#!/usr/bin/env bash
# Modal client inside a Claude Code cloud session (recipe verified in CLAUDE.md, 2026-11-05).
# Needs the cloud environment `topological-photonic-crystal-optimizer`, which allows
# api.modal.com and input-plane.us-east.modal.com. Not needed on your own laptop.
#
# - Plain `modal` cannot tunnel through HTTPS_PROXY and hangs: install modal[api-proxy-support].
# - The agent proxy re-terminates TLS: run Modal commands with SSL_CERT_FILE=/root/.ccr/ca-bundle.crt.
set -euo pipefail
V=${V:-$HOME/.venv-modal}
python3 -m venv "$V"
"$V/bin/pip" install -q "modal[api-proxy-support]" pyyaml numpy
echo "export SSL_CERT_FILE=/root/.ccr/ca-bundle.crt"
echo "export PATH=$V/bin:\$PATH"
if [ -f "$HOME/.modal.toml" ] || { [ -n "${MODAL_TOKEN_ID:-}" ] && [ -n "${MODAL_TOKEN_SECRET:-}" ]; }; then
  echo "Modal credentials: present"
else
  echo "Modal credentials: missing. Run in the background and open the printed token-flow URL:"
  echo "  SSL_CERT_FILE=/root/.ccr/ca-bundle.crt $V/bin/modal token new --source claude-code"
fi
