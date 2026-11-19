#!/bin/bash
set -euo pipefail
cd /app
git apply "$(dirname "$0")/reference.patch"
