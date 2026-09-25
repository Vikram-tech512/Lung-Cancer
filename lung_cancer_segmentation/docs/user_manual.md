# docs/user_manual.md
# LungVision AI – User Manual

## For Clinicians (Doctors, Radiologists)

### Getting Started

1. **Open the Web UI** at `http://your-server:5173`
2. You'll see the LungVision AI home page with a premium dark interface

### Running a Segmentation

1. Click **"Start Segmentation"** on the home page (or navigate to "Segment" in the navbar)
2. **Upload** your CT slice:
   - Drag & drop the image onto the upload zone, OR
   - Click the upload zone to browse your files
   - Accepted formats: PNG, JPEG (max 10 MB)
3. Once the image preview appears, click **"🔬 Run Segmentation"**
4. Wait for the progress indicator (typically < 2 seconds on CPU)
5. **View Results**:
   - Side‑by‑side view: Original CT slice | Segmentation Mask
   - The mask highlights the detected tumor region
6. **Download** the mask as a PNG by clicking "⬇ Download Mask"
7. Click "🔄 New Scan" to process another image

### History

- Navigate to **"History"** in the navbar
- View all past segmentation requests with:
  - Unique request ID
  - Filename
  - Status (completed / processing / failed)
  - Inference time
  - Timestamp

### Dashboard

- Navigate to **"Dashboard"** in the navbar
- View platform‑wide statistics:
  - Total segmentations run
  - Average Dice score
  - Average inference time
  - Number of registered users

---

## For Administrators

### Docker Deployment

```bash
# Clone the repository
git clone https://github.com/yourname/lung_cancer_segmentation.git
cd lung_cancer_segmentation

# Start all services
docker compose up -d

# Verify
curl http://localhost:8000/api/health
```

Services started:
| Service | Port | Description |
|---------|------|-------------|
| api     | 8000 | FastAPI inference + auth |
| worker  | –    | Celery async worker |
| ui      | 5173 | React front‑end |
| redis   | 6379 | Message broker |
| db      | 5432 | PostgreSQL database |

### Local Development (without Docker)

```bash
# Backend
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
python scripts/create_dummy_checkpoint.py
uvicorn src.inference.service:app --reload --host 0.0.0.0 --port 8000

# Frontend (separate terminal)
cd web
npm install
npm run dev
```

### Training a Model

```bash
python scripts/train.py \
    --img-dir data/images \
    --mask-dir data/masks \
    --epochs 50 \
    --batch-size 8 \
    --lr 1e-4 \
    --output checkpoints/vit_lung_seg.pt
```

### Evaluating a Model

```bash
python scripts/evaluate.py \
    --img-dir data/test_images \
    --mask-dir data/test_masks \
    --checkpoint checkpoints/vit_lung_seg.pt
```

### Exporting for Production

```bash
# ONNX
python scripts/export_checkpoint.py --ckpt checkpoints/vit_lung_seg.pt --format onnx

# TorchScript
python scripts/export_checkpoint.py --ckpt checkpoints/vit_lung_seg.pt --format torchscript
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Could not load checkpoint" warning | Run `python scripts/create_dummy_checkpoint.py` to create initial weights |
| API returns 500 on `/segment` | Check that the model file exists at `checkpoints/vit_lung_seg.pt` |
| UI can't connect to API | Ensure the API is running on port 8000 and CORS origins include your UI URL |
| Docker build fails | Ensure Docker Desktop is running and you have internet access for pulling base images |
