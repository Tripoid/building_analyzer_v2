#!/bin/bash
# ═══════════════════════════════════════════════════════
# Facade Analyzer — SSH Deployment Script
# Порты: 9000 (внутренний контейнер) → 44035 (внешний)
# Поддержка: systemd / Docker (без systemd)
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
echo "App port:    $APP_PORT (external: 44035)"

# ── 0. HuggingFace token (required for gated SAM3 model) ──
ENV_FILE="$PROJECT_DIR/.env"
if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
fi
if [ -z "$HF_TOKEN" ]; then
    echo ""
    echo "⚠️  HuggingFace token required to download SAM3 (gated model)."
    read -rp "  Enter HF token (hf_...): " HF_TOKEN
    echo "HF_TOKEN=$HF_TOKEN" > "$ENV_FILE"
    chmod 600 "$ENV_FILE"
    echo "  Token saved to $ENV_FILE"
fi
export HF_TOKEN

# ── 1. System dependencies ──
echo ""
echo "▶ [1/5] Installing system dependencies..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
# Core tools
apt-get install -y -qq python3 python3-pip python3-venv python3-dev curl wget git aria2
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

# SAM3.1 — unified text-prompted detection + segmentation (replaces DINO)
# Requires Python 3.12+ and PyTorch 2.7+
echo "  Installing SAM3.1..."
pip install git+https://github.com/facebookresearch/sam3.git -q
echo "  ✅ SAM3.1 installed"


# Fetch material prices from leroymerlin.ru (cached 7 days)
echo "  Fetching material prices..."
python3 price_updater.py && echo "  ✅ Prices updated" || echo "  ⚠️  Price fetch failed — using fallback prices"

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

# Kill any process holding the port (by port number, not process name)
echo "  Freeing port $APP_PORT..."
fuser -k "${APP_PORT}/tcp" 2>/dev/null || \
    lsof -ti:"${APP_PORT}" | xargs -r kill -9 2>/dev/null || true
sleep 2

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
PROJ="$(dirname "$0")"
[ -f "$PROJ/.env" ] && source "$PROJ/.env" && export HF_TOKEN
cd "$PROJ/backend"
source venv/bin/activate
exec python -m uvicorn server:app --host 0.0.0.0 --port 9000 --workers 1
STARTEOF
    chmod +x "$PROJECT_DIR/start.sh"

    # Create a stop script
    cat > "$PROJECT_DIR/stop.sh" << 'STOPEOF'
#!/bin/bash
PORT=9000
fuser -k "${PORT}/tcp" 2>/dev/null || \
    lsof -ti:"${PORT}" | xargs -r kill -9 2>/dev/null || true
echo "Server stopped (port ${PORT} freed)"
STOPEOF
    chmod +x "$PROJECT_DIR/stop.sh"

    # Start in background with nohup
    nohup bash "$PROJECT_DIR/start.sh" > "$PROJECT_DIR/server.log" 2>&1 &
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
echo "║  External: http://SERVER_IP:44035"
echo "║  API:      http://localhost:$APP_PORT/api/health   "
echo "║  Docs:     http://localhost:$APP_PORT/docs         "
echo "║                                                   ║"
echo "║  Start: $PROJECT_DIR/start.sh                      "
echo "║  Stop:  $PROJECT_DIR/stop.sh                       "
echo "║  Logs:  tail -f $PROJECT_DIR/server.log            "
echo "╚═══════════════════════════════════════════════════╝"
