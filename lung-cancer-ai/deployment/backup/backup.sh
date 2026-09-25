#!/bin/bash
# ==============================================================================
# deployment/backup/backup.sh
# Production PostgreSQL Database and Object Storage Backup
# ==============================================================================

BACKUP_DIR="${1:-/var/backups/lung-cancer-ai}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
mkdir -p "$BACKUP_DIR"

echo "[INFO] Starting database backup at $TIMESTAMP..."

# 1. PostgreSQL Database Dump
docker exec -t lung_cancer_ai_postgres pg_dump -U lungai lung_cancer_ai | gzip > "$BACKUP_DIR/db_backup_$TIMESTAMP.sql.gz"

if [ $? -eq 0 ]; then
    echo "[SUCCESS] PostgreSQL database dumped: $BACKUP_DIR/db_backup_$TIMESTAMP.sql.gz"
else
    echo "[ERROR] Database dump failed!"
    exit 1
fi

# 2. Archive Object Storage Bucket (DICOM & Masks)
echo "[INFO] Archiving object storage bucket..."
tar -czf "$BACKUP_DIR/storage_backup_$TIMESTAMP.tar.gz" -C /opt/lung-cancer-ai storage_bucket

echo "[COMPLETED] All backups completed successfully at $(date)!"
