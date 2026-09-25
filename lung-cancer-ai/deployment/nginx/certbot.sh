#!/bin/bash
# ==============================================================================
# deployment/nginx/certbot.sh
# Automated Let's Encrypt SSL Certificate Provisioning
# ==============================================================================

DOMAIN="${1:-lungcancerai.example.com}"
EMAIL="${2:-admin@lungcancerai.org}"

echo "=================================================="
echo "Acquiring SSL Certificate for: $DOMAIN"
echo "Administrative Email: $EMAIL"
echo "=================================================="

# Check if docker-compose is running
docker compose -f ../docker/docker-compose.prod.yml exec -T nginx nginx -v >/dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "[ERROR] NGINX container is not running. Please start the stack first:"
    echo "docker compose -f ../docker/docker-compose.prod.yml up -d"
    exit 1
fi

# Run certbot in docker
docker run -it --rm --name certbot \
    -v "$(pwd)/../docker/certbot_etc:/etc/letsencrypt" \
    -v "$(pwd)/../docker/certbot_var:/var/lib/letsencrypt" \
    -v "$(pwd)/../nginx:/var/www/certbot" \
    certbot/certbot certonly --webroot \
    --webroot-path=/var/www/certbot \
    --email "$EMAIL" --agree-tos --no-eff-email \
    -d "$DOMAIN"

if [ $? -eq 0 ]; then
    echo "[SUCCESS] SSL Certificate acquired successfully!"
    echo "Reloading NGINX configuration..."
    docker compose -f ../docker/docker-compose.prod.yml exec -T nginx nginx -s reload
else
    echo "[ERROR] Certbot certificate acquisition failed."
fi
