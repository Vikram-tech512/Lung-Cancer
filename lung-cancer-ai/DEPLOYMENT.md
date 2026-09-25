# 24/7 Cloud Production Server Deployment Guide

This manual guides you through deploying the **Lung Cancer AI Platform** on an independent Cloud Virtual Private Server (VPS) or dedicated server (Ubuntu 22.04 LTS / 24.04 LTS on AWS EC2, DigitalOcean Droplet, Hetzner Cloud, Google Cloud Compute Engine, or Azure VM).

Once deployed according to this guide, the platform will run **24 hours a day, 7 days a week**, completely independently of your personal laptop, browser, or terminal.

---

## 1. Cloud Server Provisioning

1. Choose a Cloud Provider (e.g. AWS EC2 `t3.xlarge` or `g4dn.xlarge` for GPU, DigitalOcean 16GB Droplet, or Hetzner CPX41).
2. Operating System: **Ubuntu 22.04 LTS x86_64**.
3. Note your public Server IPv4 Address (e.g. `159.65.120.44`).

---

## 2. Server Initial Setup & Docker Installation

SSH into your server:
```bash
ssh root@YOUR_SERVER_IP
```

Install Docker and Docker Compose:
```bash
apt-get update && apt-get upgrade -y
apt-get install -y ca-certificates curl gnupg lsb-release git

# Install Docker engine
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Enable Docker on boot
systemctl enable docker
systemctl start docker
```

---

## 3. Clone Repository & Environment Configuration

Clone your project to `/opt/lung-cancer-ai`:
```bash
mkdir -p /opt
cd /opt
git clone https://github.com/ashok910/lung-cancer-ai.git
cd /opt/lung-cancer-ai
```

Configure your environment variables:
```bash
cat << 'EOF' > .env
APP_NAME=Lung Cancer AI
PUBLIC_DOMAIN=lungcancerai.yourdomain.com
ENV=production
SECRET_KEY=generate-a-strong-random-hex-string-for-production
DATABASE_URL=postgresql+psycopg2://lungai:lungaipassword@postgres:5432/lung_cancer_ai
REDIS_URL=redis://redis:6379/0
STORAGE_ROOT=/app/storage_bucket
MAX_UPLOAD_SIZE_MB=500
EOF
```

---

## 4. Domain & DNS Configuration

1. In your Domain Registrar (Namecheap, GoDaddy, Cloudflare, Route53, etc.):
   - Add an **A Record**:
     - **Host**: `lungcancerai` (or `@` for apex domain)
     - **Points to**: `YOUR_SERVER_IP`
     - **TTL**: `300` (5 minutes)
2. Verify DNS resolution:
   ```bash
   ping lungcancerai.yourdomain.com
   ```

---

## 5. Launch Full Production Stack

```bash
cd /opt/lung-cancer-ai/deployment/docker
docker compose -f docker-compose.prod.yml up -d --build
```

Verify all 7 containers are running:
```bash
docker compose -f docker-compose.prod.yml ps
```
Expected output:
- `lung_cancer_ai_nginx`: running (ports 80, 443)
- `lung_cancer_ai_frontend`: running
- `lung_cancer_ai_backend`: running
- `lung_cancer_ai_worker`: running (Celery)
- `lung_cancer_ai_postgres`: running (healthy)
- `lung_cancer_ai_redis`: running
- `lung_cancer_ai_mlflow`: running

---

## 6. Obtain Free Let's Encrypt SSL (HTTPS)

Run the automated Certbot script:
```bash
cd /opt/lung-cancer-ai/deployment/nginx
chmod +x certbot.sh
./certbot.sh lungcancerai.yourdomain.com admin@yourdomain.com
```

Certificates will automatically renew every 90 days.

---

## 7. Enable 24/7 Systemd Daemon Auto-Restart

Ensure the entire platform starts automatically if the physical cloud server restarts or reboots:
```bash
cp /opt/lung-cancer-ai/deployment/systemd/lung-cancer-ai.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable lung-cancer-ai.service
```

---

## 8. Automated Database & Storage Backups

Set up daily automated backups via cron:
```bash
chmod +x /opt/lung-cancer-ai/deployment/backup/backup.sh
chmod +x /opt/lung-cancer-ai/deployment/backup/restore.sh

# Add to root crontab (runs every night at 2:00 AM)
(crontab -l 2>/dev/null; echo "0 2 * * * /opt/lung-cancer-ai/deployment/backup/backup.sh /var/backups/lung-cancer-ai >> /var/log/backup.log 2>&1") | crontab -
```

---

## 9. Final 24/7 Acceptance Test

1. Turn off your personal laptop or close your IDE completely.
2. Open your smartphone or a completely different computer connected to 4G/5G mobile data.
3. Navigate to: `https://lungcancerai.yourdomain.com`
4. Confirm:
   - Green SSL padlock is active.
   - Public Landing Page loads.
   - Login works with `admin@lungcancerai.org`.
   - CT Workstation renders interactive 3D overlays and EKG confidence signature.
   - Background Celery pipeline executes and delivers real-time updates over WebSocket.
