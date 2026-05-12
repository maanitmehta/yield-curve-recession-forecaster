#!/usr/bin/env bash
# =============================================================
#  NS Yield Curve Project — one-shot environment setup
#  Run from the project root:  bash setup_env.sh
# =============================================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

echo "=============================================="
echo " NS Yield Curve Project — Environment Setup"
echo "=============================================="
echo "Project : $PROJECT_DIR"
echo "Venv    : $VENV_DIR"
echo ""

# 1. Create virtual environment
python3 -m venv "$VENV_DIR"
echo "[1/4] Virtual environment created"

# 2. Activate
source "$VENV_DIR/bin/activate"
echo "[2/4] Activated .venv"

# 3. Upgrade pip silently
pip install --upgrade pip --quiet

# 4. Install requirements
pip install -r "$PROJECT_DIR/requirements.txt"
echo "[3/4] Dependencies installed"

# 5. Create folder structure
mkdir -p "$PROJECT_DIR/data"
mkdir -p "$PROJECT_DIR/output"
echo "[4/4] Folders created: data/  output/"

echo ""
echo "=============================================="
echo " Setup complete."
echo ""
echo " To activate the environment in future:"
echo "   source .venv/bin/activate"
echo ""
echo " Then run Phase 1:"
echo "   python phase1_data.py"
echo ""
echo " Then run Phase 2:"
echo "   python phase2_nelson_siegel.py"
echo "=============================================="
