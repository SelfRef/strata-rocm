#!/bin/sh
# strata-rocm entrypoint
#
#   prepare <setup.py args>        one-time model preparation into /data (pack, MTP draft layer, server config)
#   serve <model> [server args]    start the OpenAI-compatible server from /data/strata-<model>.json
#   anything else                  run as a command (e.g. bash)
set -eu
cd /opt/strata

case "${1:-help}" in
  prepare)
    shift
    .venv/bin/python setup.py --backend hip --data-dir /data --host 0.0.0.0 \
      --yes --no-start --no-browser "$@"
    # setup.py writes the server config next to itself, inside the image: keep it in the volume
    cp -f strata-*.json /data/
    ls /data/strata-*.json
    ;;
  serve)
    shift
    [ $# -ge 1 ] || { echo "usage: serve <model> [--port N] [server args]" >&2; exit 2; }
    model=$(echo "$1" | tr 'A-Z' 'a-z'); shift
    cfg=/data/strata-$model.json
    [ -f "$cfg" ] || { echo "$cfg not found: run 'prepare' first" >&2; exit 1; }
    exec .venv/bin/python -m serve.server --engine strata --config "$cfg" --host 0.0.0.0 "$@"
    ;;
  help|-h|--help)
    sed -n '2,7p' "$0"
    ;;
  *)
    exec "$@"
    ;;
esac
