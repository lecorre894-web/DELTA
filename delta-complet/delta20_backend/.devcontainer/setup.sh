#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update
sudo apt-get install -y build-essential git curl jq htop

# --- Environnement Python quantique ---
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install "qiskit[visualization]" qiskit-ibm-runtime qiskit-aer

# --- Credentials IBM Quantum (à remplir une fois) ---
if [ ! -f .env ]; then
  cat > .env <<'EOF2'
DELTA_ENV=development
DELTA_PORT=8000
IBM_QUANTUM_TOKEN=
IBM_QUANTUM_INSTANCE=ibm-q/open/main
EOF2
  echo "✅ .env créé — renseigne IBM_QUANTUM_TOKEN"
fi

echo "🚀 Environnement quantique DELTA prêt !"
