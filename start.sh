#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# start.sh — Insight Agent startup script
#
# - Checks for an existing .venv, creates one with Python 3.12 if absent
# - Warns if the Python version is above 3.12
# - Installs dependencies from requirements.txt (if present)
# - Starts the FastAPI server on 0.0.0.0:8000
# - Opens http://localhost:8000 in the default browser
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
REQUIRED_MAJOR=3
REQUIRED_MINOR=12

# ── Helper: print a section header ───────────────────────────────────────────
header() { echo -e "\n${BOLD}${CYAN}▸ $1${RESET}"; }
ok()     { echo -e "  ${GREEN}✓${RESET}  $1"; }
warn()   { echo -e "  ${YELLOW}⚠${RESET}  $1"; }
err()    { echo -e "  ${RED}✗${RESET}  $1"; }

# ─────────────────────────────────────────────────────────────────────────────
# 1. Locate Python 3.12
# ─────────────────────────────────────────────────────────────────────────────
header "Locating Python $REQUIRED_MAJOR.$REQUIRED_MINOR"

PYTHON_BIN=""

# Prefer an explicit python3.12 binary
if command -v python3.12 &>/dev/null; then
    PYTHON_BIN="$(command -v python3.12)"
else
    # Fall back to whatever python3 / python resolves to
    for candidate in python3 python; do
        if command -v "$candidate" &>/dev/null; then
            PYTHON_BIN="$(command -v "$candidate")"
            break
        fi
    done
fi

if [[ -z "$PYTHON_BIN" ]]; then
    err "No Python interpreter found. Please install Python 3.12."
    exit 1
fi

# Read the actual version
PY_VERSION=$("$PYTHON_BIN" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
PY_MAJOR=$("$PYTHON_BIN" -c "import sys; print(sys.version_info.major)")
PY_MINOR=$("$PYTHON_BIN" -c "import sys; print(sys.version_info.minor)")

if [[ "$PY_MAJOR" -ne "$REQUIRED_MAJOR" || "$PY_MINOR" -lt "$REQUIRED_MINOR" ]]; then
    err "Python $PY_VERSION found at $PYTHON_BIN — need $REQUIRED_MAJOR.$REQUIRED_MINOR+."
    exit 1
fi

if [[ "$PY_MINOR" -gt "$REQUIRED_MINOR" ]]; then
    warn "Python $PY_VERSION is above $REQUIRED_MAJOR.$REQUIRED_MINOR."
    warn "This project was tested on Python 3.12. Higher versions may have compatibility issues."
    echo ""
    read -r -p "  Continue anyway? [y/N] " REPLY
    if [[ ! "$REPLY" =~ ^[Yy]$ ]]; then
        echo "  Aborted."
        exit 1
    fi
else
    ok "Python $PY_VERSION found at $PYTHON_BIN"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 2. Create / reuse virtual environment
# ─────────────────────────────────────────────────────────────────────────────
header "Virtual environment"

VENV_EXISTED=false
if [[ -d "$VENV_DIR" && -f "$VENV_DIR/bin/activate" ]]; then
    ok "Existing venv found at $VENV_DIR"
    VENV_EXISTED=true
else
    echo "  Creating venv at $VENV_DIR …"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    ok "venv created"
fi

# Activate
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
ok "venv activated  ($(python --version))"

# ─────────────────────────────────────────────────────────────────────────────
# 3. Install / upgrade dependencies (only on fresh venv)
# ─────────────────────────────────────────────────────────────────────────────
header "Dependencies"

if [[ "$VENV_EXISTED" == true ]]; then
    ok "Existing venv — skipping dependency install"
else
    pip install --upgrade pip --quiet

    if [[ -f "$SCRIPT_DIR/requirements.txt" ]]; then
        echo "  Installing from requirements.txt …"
        pip install -r "$SCRIPT_DIR/requirements.txt" --quiet
        ok "Dependencies installed"
    else
        warn "requirements.txt not found — skipping dependency install."
        warn "Make sure all packages are already present in the venv."
    fi

    # ── MinerU — must be installed via uv, not pip ────────────────────────────
    echo "  Installing MinerU via uv …"
    if ! command -v uv &>/dev/null; then
        warn "uv not found — installing it now …"
        pip install uv --quiet
    fi
    uv pip install -U "mineru[all]" --quiet
    ok "MinerU installed"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 4. Verify server.py exists
# ─────────────────────────────────────────────────────────────────────────────
header "Pre-flight checks"

if [[ ! -f "$SCRIPT_DIR/server.py" ]]; then
    err "server.py not found in $SCRIPT_DIR"
    exit 1
fi
ok "server.py found"

# Ensure data directories exist (server expects them at startup)
mkdir -p "$SCRIPT_DIR/data/docs/inputs"
mkdir -p "$SCRIPT_DIR/data/docs/outputs"
mkdir -p "$SCRIPT_DIR/data/chroma_db"
ok "Data directories ready"

# ─────────────────────────────────────────────────────────────────────────────
# 5. Open browser once server is ready
# ─────────────────────────────────────────────────────────────────────────────
open_browser() {
    local url="http://localhost:8000"
    echo -e "\n  ${CYAN}Waiting for server to become ready…${RESET}"
    until curl -s --max-time 1 "$url/health" &>/dev/null; do
        sleep 0.5
    done
    ok "Server ready — opening $url"
    if command -v xdg-open &>/dev/null; then
        xdg-open "$url" &>/dev/null &
    elif command -v open &>/dev/null; then
        open "$url" &
    else
        warn "Could not detect a browser opener. Visit $url manually."
    fi
}

# Launch browser probe in the background so it doesn't block uvicorn
open_browser &

# ─────────────────────────────────────────────────────────────────────────────
# 6. Start FastAPI server
# ─────────────────────────────────────────────────────────────────────────────
header "Starting Insight Agent server"
echo -e "  ${BOLD}http://0.0.0.0:8000${RESET}  (Ctrl-C to stop)\n"

cd "$SCRIPT_DIR"
exec uvicorn server:app --host 0.0.0.0 --port 8000 --reload