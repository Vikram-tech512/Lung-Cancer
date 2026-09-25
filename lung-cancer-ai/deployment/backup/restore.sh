#!/bin/bash
# ==============================================================================
# deployment/backup/restore.sh
# Production Restoration Script for PostgreSQL and Storage
# ==============================================================================

if [ -z "$1" ]; then
    echo "Usage: ./restore.sh <path_to_db_backup.sql.gz> [path_to_storage_backup.tar.gz]"
    exit 1
fi

DB_BACKUP_FILE="$1"
STORAGE_BACKUP_FILE="$2"

echo "[WARNING] This will restore the database from $DB_BACKUP_FILE and overwrite current data!"
read -p "Are you sure you want to proceed? (y/N): " confirm
if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "Restore cancelled."
    exit 0
fi

echo "[INFO] Restoring database..."
gunzip < "$DB_BACKUP_FILE" | docker exec -i lung_cancer_ai_postgres psql -U lungai -d lung_cancer_ai

if [ $? -eq 0 ]; then
    echo "[SUCCESS] Database restoration complete."
else
    echo "[ERROR] Database restoration failed!"
    exit 1
fi

if [ -n "$STORAGE_BACKUP_FILE" ] && [ -f "$STORAGE_BACKUP_FILE" ]; then
    echo "[INFO] Restoring object storage..."
    tar -xzf "$STORAGE_BACKUP_FILE" -C /opt/lung-cancer-ai/
    echo "[SUCCESS] Object storage restored."
fi

echo "[COMPLETED] System restoration finished successfully."
