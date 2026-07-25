set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Expose the venv-local Lean toolchain to the compiler reward.
export ELAN_HOME="$SCRIPT_DIR/../venv/elan"
export PATH="$ELAN_HOME/bin:$PATH"

venv/bin/python rlcf/main.py "$@"
