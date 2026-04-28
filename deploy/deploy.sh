#!/bin/bash
# ═══════════════════════════════════════════════════════
# Facade Analyzer — SSH Deployment Script
# Порты: 8080 → 20090 (qudata.ai маппинг)
# Поддержка: systemd / Docker (без systemd)
# ═══════════════════════════════════════════════════════
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
APP_PORT=8080

echo "╔═══════════════════════════════════════════════════╗"
echo "║  🏗️  Facade Analyzer — Deployment                ║"
echo "╚═══════════════════════════════════════════════════╝"
echo ""
echo "Project dir: $PROJECT_DIR"
echo "App port:    $APP_PORT (external: 20090)"

# ── 1. System dependencies ──
echo ""
echo "▶ [1/5] Installing system dependencies..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
# Core tools
apt-get install -y -qq python3 python3-pip python3-venv python3-dev curl wget git
# OpenCV runtime deps (Ubuntu 24.04 Noble — libgl1 replaces libgl1-mesa-glx)
apt-get install -y -qq \
    libgl1 libglib2.0-0t64 libsm6 libxext6 libxrender1 \
    libxcb1 libxcb-shm0 libxcb-xfixes0 \
    libfontconfig1 libice6 \
    libgomp1 \
    || true

# Install Node.js (LTS) if missing
if ! command -v node &> /dev/null; then
    echo "  Installing Node.js..."
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
    apt-get install -y -qq nodejs
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
pip install git+https://github.com/facebookresearch/sam2.git -q 2>/dev/null || echo "  ⚠️ SAM2 install skipped (optional)"

# Download SAM2 weights if not present
if [ ! -f sam2_hiera_small.pt ]; then
    echo "  Downloading SAM2 weights..."
    wget -q https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt || echo "  ⚠️ SAM2 weights download failed (optional)"
fi
echo "  ✅ Python environment ready"

# ── 3. Build React frontend ──
echo ""
echo "▶ [3/5] Building React frontend..."
cd "$PROJECT_DIR/frontend"
npm install --silent
npm run build
echo "  ✅ Frontend built → frontend/dist/"

# ── 4. Start server ──
echo ""
echo "▶ [4/5] Starting server..."

# Kill any existing instance
pkill -f "uvicorn server:app.*--port $APP_PORT" 2>/dev/null || true
sleep 1

cd "$PROJECT_DIR/backend"
source venv/bin/activate

# Check if systemd is available
if command -v systemctl &> /dev/null && [ -d /run/systemd/system ]; then
    echo "  Using systemd..."
    SERVICE_FILE="$SCRIPT_DIR/facade-analyzer.service"
    sed -i "s|/opt/facade-analyzer|$PROJECT_DIR|g" "$SERVICE_FILE" 2>/dev/null || true
    cp "$SERVICE_FILE" /etc/systemd/system/facade-analyzer.service
    systemctl daemon-reload
    systemctl enable facade-analyzer
    systemctl restart facade-analyzer
    echo "  ✅ Systemd service started"
else
    echo "  No systemd detected (Docker container). Using nohup..."
    # Create a start script
    cat > "$PROJECT_DIR/start.sh" << 'STARTEOF'
#!/bin/bash
cd "$(dirname "$0")/backend"
source venv/bin/activate
exec python -m uvicorn server:app --host 0.0.0.0 --port 8080 --workers 1
STARTEOF
    chmod +x "$PROJECT_DIR/start.sh"

    # Create a stop script
    cat > "$PROJECT_DIR/stop.sh" << 'STOPEOF'
#!/bin/bash
pkill -f "uvicorn server:app.*--port 8080" 2>/dev/null && echo "Server stopped" || echo "Server was not running"
STOPEOF
    chmod +x "$PROJECT_DIR/stop.sh"

    # Start in background with nohup
    nohup "$PROJECT_DIR/start.sh" > "$PROJECT_DIR/server.log" 2>&1 &
    SERVER_PID=$!
    echo "  ✅ Server started (PID: $SERVER_PID)"
    echo "  📄 Logs: tail -f $PROJECT_DIR/server.log"
fi

# ── 5. Verify ──
echo ""
echo "▶ [5/5] Verifying..."
sleep 5
if curl -s http://localhost:$APP_PORT/api/health 2>/dev/null | grep -q '"status"'; then
    echo "  ✅ Server is responding!"
else
    echo "  ⏳ Server is starting (ML models loading, may take 1-5 min)..."
    echo "  Check: tail -f $PROJECT_DIR/server.log"
fi

# ── Done ──
SERVER_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "localhost")
echo ""
echo "╔═══════════════════════════════════════════════════╗"
echo "║  🚀 DEPLOYMENT COMPLETE!                         ║"
echo "╠═══════════════════════════════════════════════════╣"
echo "║                                                   ║"
echo "║  Local:    http://localhost:$APP_PORT              "
echo "║  External: http://SERVER_IP:20090"
echo "║  API:      http://localhost:$APP_PORT/api/health   "
echo "║  Docs:     http://localhost:$APP_PORT/docs         "
echo "║                                                   ║"
echo "║  Start: $PROJECT_DIR/start.sh                      "
echo "║  Stop:  $PROJECT_DIR/stop.sh                       "
echo "║  Logs:  tail -f $PROJECT_DIR/server.log            "
echo "╚═══════════════════════════════════════════════════╝"
