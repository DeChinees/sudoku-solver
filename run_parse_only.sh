#!/usr/bin/env bash
set -euo pipefail

# -------- config --------
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
SCRIPT="${SCRIPT:-parse_only.py}"
IMAGE="${1:-puzzle01.png}"

# -------- helpers --------
have_cmd() { command -v "$1" >/dev/null 2>&1; }

print_tesseract_help() {
  cat <<'EOS'
[!] Tesseract OCR is not installed (command 'tesseract' not found).
    Please install it using one of the following:

  macOS (Homebrew):
    brew install tesseract

  Ubuntu / Debian:
    sudo apt-get update && sudo apt-get install -y tesseract-ocr

  Fedora:
    sudo dnf install tesseract

  Arch / Manjaro:
    sudo pacman -S tesseract

  openSUSE:
    sudo zypper install tesseract-ocr

  After installing, re-run this script.
EOS
}

# -------- prechecks --------
if [[ ! -f "$SCRIPT" ]]; then
  echo "[x] Can't find $SCRIPT in $(pwd)"
  echo "    Make sure parse_only.py is saved here (the OCR reader we discussed)."
  exit 1
fi

if ! have_cmd tesseract; then
  print_tesseract_help
  exit 2
fi

if ! have_cmd "$PYTHON_BIN"; then
  echo "[x] '$PYTHON_BIN' not found. Install Python 3.x and re-run."
  exit 3
fi

# -------- venv setup & run (in subshell so we don't pollute current shell) --------
(
  if [[ ! -d "$VENV_DIR" ]]; then
    echo "[*] Creating virtual environment at $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  else
    echo "[*] Using existing virtual environment $VENV_DIR"
  fi

  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"

  echo "[*] Upgrading pip"
  python -m pip install --upgrade pip >/dev/null

  echo "[*] Installing Python dependencies (opencv-python, pytesseract, numpy)…"
  python -m pip install --quiet opencv-python pytesseract numpy

  echo "[*] Running: python $SCRIPT $IMAGE"
  python "$SCRIPT" "$IMAGE"
)

echo "[*] Done. (venv was used temporarily and is not active in your shell)"