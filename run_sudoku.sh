#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_sudoku.sh --image puzzle.jpg --out solved.png
#
# This script will:
#   1. Check if Tesseract OCR is installed, and show install instructions if missing.
#   2. Create a .venv if not present.
#   3. Install Python deps from requirements.txt.
#   4. Run sudoku_from_image.py with the provided arguments.

VENV=".venv"

check_tesseract() {
    if command -v tesseract >/dev/null 2>&1; then
        echo "[*] Tesseract found: $(command -v tesseract)"
        return 0
    fi

    echo "[!] Tesseract is not installed on this system."
    case "$(uname -s)" in
        Linux*)
            echo "    Install it using your package manager, e.g.:"
            echo "      Debian/Ubuntu:   sudo apt update && sudo apt install -y tesseract-ocr"
            echo "      Fedora:          sudo dnf install -y tesseract"
            echo "      Arch Linux:      sudo pacman -S tesseract"
            ;;
        Darwin*)
            echo "    On macOS, install it with Homebrew:"
            echo "      brew install tesseract"
            ;;
        *)
            echo "    Unsupported OS detected. Please install Tesseract manually."
            ;;
    esac
    echo "    After installing, re-run this script."
    exit 1
}

setup_venv() {
    if [ ! -d "$VENV" ]; then
        echo "[*] Creating virtual environment in $VENV..."
        python3 -m venv "$VENV"
        echo "[*] Upgrading pip..."
        "$VENV/bin/pip" install --upgrade pip
        echo "[*] Installing requirements..."
        "$VENV/bin/pip" install -r requirements.txt
    else
        echo "[*] Using existing virtual environment $VENV"
    fi
}

main() {
    check_tesseract
    setup_venv

    echo "[*] Running sudoku_from_image.py with args: $@"
    exec "$VENV/bin/python" sudoku_from_image.py "$@"
}

main "$@"