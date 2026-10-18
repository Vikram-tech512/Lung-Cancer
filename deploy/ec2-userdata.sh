#!/bin/bash
# ============================================================
# EC2 User Data Script — Lung Cancer AI Platform Backend
# Runs automatically on first EC2 instance launch.
# Installs Docker, clones the repo, and starts the backend.
# ============================================================

set -e
exec > /var/log/user-data.log 2>&1

echo "=========================================="
echo "  Lung Cancer AI — EC2 Bootstrap"
echo "=========================================="

# 1. Update system & install dependencies
yum update -y
yum install -y docker git curl

# 2. Start & enable Docker
systemctl start docker
systemctl enable docker
usermod -aG docker ec2-user

# 3. Install Docker Compose v2
mkdir -p /usr/local/lib/docker/cli-plugins
curl -SL "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64" \
  -o /usr/local/lib/docker/cli-plugins/docker-compose
chmod +x /usr/local/lib/docker/cli-plugins/docker-compose

# 4. Clone the repository
# Replace <YOUR_GITHUB_USERNAME>/<YOUR_REPO_NAME> with actual repo
git clone https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPO_NAME>.git /app
cd /app

# 5. Write .env file
cat > /app/.env <<EOF
PORT=8000
HOST=0.0.0.0
ENVIRONMENT=production
DATABASE_URL=sqlite:///./lung_platform.db
LOG_LEVEL=INFO
ANTHROPIC_API_KEY=
MLFLOW_TRACKING_URI=./experiments_log
EOF

# 6. Build & run only the backend service (frontend served via S3+CloudFront)
docker compose up -d --build backend

# 7. Create a systemd service to auto-restart on reboot
cat > /etc/systemd/system/lung-cancer-backend.service <<EOF
[Unit]
Description=Lung Cancer AI Backend
After=docker.service
Requires=docker.service

[Service]
WorkingDirectory=/app
ExecStart=/usr/local/lib/docker/cli-plugins/docker-compose up backend
ExecStop=/usr/local/lib/docker/cli-plugins/docker-compose stop backend
Restart=always
RestartSec=10
User=root

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable lung-cancer-backend

echo "=========================================="
echo "  Bootstrap complete! Backend is running."
PUB_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)
echo "  Health: http://${PUB_IP}:8000/api/v2/health"
echo "=========================================="
