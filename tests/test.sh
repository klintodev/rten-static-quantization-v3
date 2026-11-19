#!/bin/bash
set -euo pipefail
mkdir -p /logs/verifier
echo 0 > /logs/verifier/reward.txt
cd /app
export CARGO_NET_OFFLINE=true
# Restored submissions can predate the binaries cached in the development image.
# Compile the verifier and submitted crates together from the restored source.
CARGO_TARGET_DIR=$(mktemp -d /tmp/rten-verifier-target.XXXXXX)
export CARGO_TARGET_DIR
if python3 /tests/verify.py; then
    echo 1 > /logs/verifier/reward.txt
fi
