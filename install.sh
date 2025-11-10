#!/usr/bin/env bash
echo "Installing dependencies and activating virtualenv..."

set -e

echo "(re)setting virtual environment"
rm -rf .venv/
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

mkdir -p logs

echo "Installation complete!"
