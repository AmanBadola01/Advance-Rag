#!/usr/bin/env bash
set -eu
APP_ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
export PM2_HOME="$APP_ROOT/deploy/.pm2"
cd "$APP_ROOT"
# Foreground PM2 avoids daemon startup hangs in the restricted runtime.
exec pm2 start deploy/ecosystem.config.cjs --no-daemon
