# 🏗️ Facade Analyzer — Деплой по SSH

## Архитектура

```
┌──────────────────────────────────────────────────┐
│              Сервер                              │
│                                                  │
│  ┌────────────────────────────────────────────┐  │
│  │  FastAPI (uvicorn) :9000                   │  │
│  │  ├── /api/*     → ML Pipeline + REST API   │  │
│  │  └── /*         → React SPA (static)       │  │
│  └────────────────────────────────────────────┘  │
│                                                  │
│  Внешний доступ:  :44025 → :9000                 │
└──────────────────────────────────────────────────┘
```

## Порты сервера

| Внешний порт | Внутренний порт | Назначение         |
|:---:|:---:|---|
| 44023 | 22   | SSH                        |
| 44024 | 8888 | Jupyter                    |
| **44025** | **9000** | **Facade Analyzer (web)** |
| 44026–44032 | 9001–9007 | Свободны            |

## Требования к серверу

- **OS**: Ubuntu 20.04+ / Debian 11+
- **GPU**: NVIDIA с CUDA (рекомендуется, для ML)
- **RAM**: минимум 8 GB (16 GB рекомендуется)
- **Диск**: 20 GB+

## Быстрый деплой (одна команда)

### 1. Подключитесь к серверу

```bash
ssh -p 44023 user@YOUR_SERVER_IP
```

### 2. Клонируйте репозиторий

```bash
cd /opt
git clone https://github.com/Tripoid/building_analyzer_v2.git facade-analyzer
cd facade-analyzer
```

### 3. Запустите деплой

```bash
chmod +x deploy/deploy.sh
sudo ./deploy/deploy.sh
```

Скрипт автоматически:
- ✅ Установит Python, Node.js
- ✅ Создаст Python venv и установит ML-зависимости
- ✅ Соберёт React-фронтенд
- ✅ Настроит systemd-сервис на порту 9000
- ✅ Скачает веса SAM2

### 4. Откройте в браузере

```
http://YOUR_SERVER_IP:44025
```

## Ручной деплой (пошагово)

### Backend

```bash
cd /opt/facade-analyzer/backend

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/sam2.git

# Скачать веса SAM2
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt

# Тест (порт 9000)
python server.py
```

### Frontend

```bash
cd /opt/facade-analyzer/frontend
npm install
npm run build
```

### Systemd

```bash
sudo cp deploy/facade-analyzer.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable facade-analyzer
sudo systemctl start facade-analyzer
```

## Управление

```bash
# Статус
sudo systemctl status facade-analyzer

# Логи (live)
sudo journalctl -u facade-analyzer -f

# Перезапуск
sudo systemctl restart facade-analyzer

# Остановка
sudo systemctl stop facade-analyzer
```

## Обновление

```bash
cd /opt/facade-analyzer
git pull

cd frontend && npm run build && cd ..
sudo systemctl restart facade-analyzer
```

## Структура проекта

```
building_analyzer_v2/
├── backend/
│   ├── server.py              # FastAPI :9000 + React static
│   ├── ml_pipeline.py         # ML: DINO + SAM + CLIPSeg + SAM2
│   ├── repair_calculator.py   # Калькулятор стоимости ремонта
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── pages/             # React: Home, Upload, Loading, Results, Settings
│   │   ├── components/        # StatCard, DamageChart, CostBreakdownCard
│   │   ├── api/               # API клиент
│   │   └── index.css          # Дизайн-система
│   └── dist/                  # Production build
├── deploy/
│   ├── deploy.sh              # Автодеплой
│   ├── facade-analyzer.service # Systemd (порт 9000)
│   └── nginx.conf             # Опционально (если нужен reverse proxy)
└── DEPLOY.md
```
