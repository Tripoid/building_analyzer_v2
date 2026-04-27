#!/bin/bash
# ═══════════════════════════════════════════════════════
# Facade Analyzer — SSH Deployment Script
# Порты сервера: 44025 → 9000 (приложение)
# ═══════════════════════════════════════════════════════
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
APP_PORT=9000

echo "╔═══════════════════════════════════════════════════╗"
echo "║  🏗️  Facade Analyzer — Deployment                ║"
echo "╚═══════════════════════════════════════════════════╝"
echo ""
echo "Project dir: $PROJECT_DIR"
echo "App port:    $APP_PORT (external: 44025)"

# ── 1. System dependencies ──
echo ""
echo "▶ [1/5] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3 python3-pip python3-venv curl

# Install Node.js (LTS) if missing
if ! command -v node &> /dev/null; then
    echo "  Installing Node.js..."
    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
    sudo apt-get install -y -qq nodejs
fi
echo "  ✅ System deps ready"

# ── 2. Python virtual environment ──
echo ""
echo "▶ [2/5] Setting up Python venv..."
cd "$PROJECT_DIR/backend"
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
pip install git+https://github.com/facebookresearch/sam2.git -q

# Download SAM2 weights if not present
if [ ! -f sam2_hiera_small.pt ]; then
    echo "  Downloading SAM2 weights..."
    wget -q https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt
fi
echo "  ✅ Python environment ready"

# ── 3. Build React frontend ──
echo ""
echo "▶ [3/5] Building React frontend..."
cd "$PROJECT_DIR/frontend"
npm install --silent
npm run build
echo "  ✅ Frontend built → frontend/dist/"

# ── 4. Setup systemd service ──
echo ""
echo "▶ [4/5] Setting up systemd service..."
SERVICE_FILE="$SCRIPT_DIR/facade-analyzer.service"

# Update paths in service file to actual project dir
sed -i "s|/opt/facade-analyzer|$PROJECT_DIR|g" "$SERVICE_FILE" 2>/dev/null || true

sudo cp "$SERVICE_FILE" /etc/systemd/system/facade-analyzer.service
sudo systemctl daemon-reload
sudo systemctl enable facade-analyzer
sudo systemctl restart facade-analyzer
echo "  ✅ Service installed and started"

# ── 5. Verify ──
echo ""
echo "▶ [5/5] Verifying..."
sleep 3
if curl -s http://localhost:$APP_PORT/api/health | grep -q '"status"'; then
    echo "  ✅ Server is responding!"
else
    echo "  ⏳ Server is starting (ML models loading, may take 1-2 min)..."
    echo "  Check: sudo journalctl -u facade-analyzer -f"
fi

# ── Done ──
SERVER_IP=$(hostname -I | awk '{print $1}')
echo ""
echo "╔═══════════════════════════════════════════════════╗"
echo "║  🚀 DEPLOYMENT COMPLETE!                         ║"
echo "╠═══════════════════════════════════════════════════╣"
echo "║                                                   ║"
echo "║  Local:    http://localhost:$APP_PORT              "
echo "║  External: http://SERVER_IP:44025                  "
echo "║  API:      http://localhost:$APP_PORT/api/health   "
echo "║  Docs:     http://localhost:$APP_PORT/docs         "
echo "║                                                   ║"
echo "║  Logs:   sudo journalctl -u facade-analyzer -f    ║"
echo "║  Status: sudo systemctl status facade-analyzer    ║"
echo "╚═══════════════════════════════════════════════════╝"
