#!/bin/bash
# Launch script for Syntax Tuning (Stage 1 SFT).
# Usage:  bash main.sh <run_name>
# Example: bash main.sh test1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

venv/bin/python main.py "$@"
