#!/bin/sh
# DOCBOT-2338: Railway mounts volumes root-owned. Start as root only to hand
# /data to appuser, then drop privileges for the real process.
set -e
if [ "$(id -u)" = "0" ]; then
  # Only touch entries not already owned by appuser, so restarts stay fast.
  [ -d /data ] && find /data \! -user appuser -exec chown appuser:appuser {} +
  export HOME=/home/appuser
  exec setpriv --reuid=appuser --regid=appuser --init-groups -- "$@"
fi
# Already non-root (e.g. a uid was forced): nothing to chown, just run.
exec "$@"
