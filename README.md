#Crowd Detection System

A Flask web application that counts people in images using a remote detection API deployed on Hugging Face Spaces. The Flask app sends images to the model API, which runs YOLOv8 + YuNet detection and returns annotated images with head counts. Annotated images are stored permanently on Cloudinary, and all records are persisted in MongoDB Atlas — surviving across deployments.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Deploying the Model API on Hugging Face Spaces](#deploying-the-model-api-on-hugging-face-spaces)
- [Local Development Setup](#local-development-setup)
- [Environment Variables](#environment-variables)
- [Deploying the Web App to Render](#deploying-the-web-app-to-render)
- [Admin Panel](#admin-panel)
- [API Reference](#api-reference)
- [Notes](#notes)

---

## Architecture Overview

Community uses a **two-service architecture**:

```
┌─────────────────────────┐         ┌──────────────────────────────────┐
│   Flask Web App         │         │   Model API (HF Spaces)          │
│   (Render / local)      │         │   (Hugging Face Spaces)          │
│                         │  POST   │                                  │
│  User uploads image ──────────▶  │  YOLOv8m body detection          │
│                         │ /detect │  YuNet face detection            │
│  Receives annotated  ◀──────────  │  Merge + annotate + return       │
│  image + head count     │  JSON   │                                  │
│                         │         │  Returns: head_count +           │
│  Stores to Cloudinary   │         │  base64 annotated JPEG           │
│  + MongoDB Atlas        │         │                                  │
└─────────────────────────┘         └──────────────────────────────────┘
```

- **Model API** — a standalone Python service on Hugging Face Spaces that loads YOLOv8m + YuNet, performs detection, and returns results via HTTP.
- **Web App** — this Flask app. It does NOT run any ML models locally. It sends images to the Model API, receives results, and handles the UI, storage, and admin panel.

---

## Features

- **Remote detection engine** — sends images to a Hugging Face Spaces model API running YOLOv8m (body detection) + YuNet (face detection)
- **Upload or capture** — users can upload an image file or take a photo directly from their device camera
- **Annotated output** — bounding boxes, confidence badges, and a people-count banner are drawn on every processed image (by the model API)
- **Cloud storage** — annotated images uploaded to Cloudinary CDN; URLs stored permanently in MongoDB Atlas
- **Admin dashboard** — analytics (totals, averages, per-event stats, daily chart), photo gallery with lightbox, place management, record editing
- **Place management** — create, rename, and delete places/events from the admin panel
- **Persistent across deploys** — MongoDB Atlas stores all records and events; Cloudinary stores all images

---

## Tech Stack

| Layer | Technology |
|---|---|
| Web framework | Flask 3.0 |
| Production server | Gunicorn 22 |
| Detection API | Hugging Face Spaces (YOLOv8m + YuNet) |
| HTTP client | Requests |
| Database | MongoDB Atlas (pymongo) |
| Image CDN | Cloudinary |
| Deployment (Web App) | Render (Docker) |
| Deployment (Model) | Hugging Face Spaces (Docker) |

---

## Project Structure

```
crowd_detection2/
│
├── app.py                  # Flask routes and application logic
├── detection.py            # HTTP client — sends images to HF model API
├── db.py                   # MongoDB Atlas access layer
├── config.py               # App configuration and constants
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container definition for Render
│
├── static/
│   ├── css/style.css       # Application styles
│   └── js/
│       ├── camera.js       # Camera capture & upload logic
│       └── dialogs.js      # Custom alert/confirm dialogs
│
├── templates/
│   ├── base.html           # Base layout with navbar
│   ├── upload.html         # Public upload page
│   └── admin/
│       ├── login.html      # Admin login
│       ├── dashboard.html  # Analytics dashboard
│       ├── photos.html     # Photo gallery with lightbox
│       ├── places.html     # Place management
│       └── edit_record.html
│
└── uploads/                # Temporary folder (files deleted after detection)
```

---

## How It Works

### Detection Pipeline

```
User uploads image
       │
       ▼
Saved temporarily to disk (uploads/)
       │
       ▼
Image sent via HTTP POST to HF Spaces Model API (/detect)
       │
       ▼
Model API runs YOLOv8m (body detection) + YuNet (face detection)
       │
       ▼
Model API merges results, draws annotations, returns JSON:
  { "head_count": N, "annotated_image": "<base64 JPEG>" }
       │
       ▼
Flask app decodes base64 annotated image
       │
       ▼
Delete original temp file from disk
       │
       ▼
Upload annotated bytes → Cloudinary (permanent CDN URL)
       │
       ▼
Save record (event, count, Cloudinary URL, timestamp) → MongoDB Atlas
```

---

## Deploying the Model API on Hugging Face Spaces

This section walks you through creating a **standalone model API** on Hugging Face Spaces that runs YOLOv8 + YuNet detection and exposes an HTTP endpoint.

### Prerequisites

- A free [Hugging Face](https://huggingface.co) account
- Your model files:
  - `yolov8m.pt` (YOLOv8 medium — body detection)
  - `face_detection_yunet_2023mar.onnx` (YuNet — face detection)

### Step 1: Create a New Hugging Face Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Fill in the form:
   - **Owner**: your HF username (e.g., `mourishantony`)
   - **Space name**: `crowd-detection-model`
   - **License**: choose one (e.g., MIT)
   - **SDK**: select **Docker**
   - **Visibility**: Public (or Private if you prefer)
3. Click **Create Space**

### Step 2: Clone the Space Locally

```bash
git clone https://huggingface.co/spaces/<your-username>/crowd-detection-model
cd crowd-detection-model
```

> Replace `<your-username>` with your actual Hugging Face username.

### Step 3: Create the Model API Files

You need to create **4 files** inside the cloned Space folder:

#### 3a. `app.py` — FastAPI server with detection logic

```python
import base64
import io
import os
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO

app = FastAPI()

# ── Load models once at startup ──────────────────────────────────────

YOLO_MODEL_PATH = "yolov8m.pt"
YUNET_MODEL_PATH = "face_detection_yunet_2023mar.onnx"

_yolo = YOLO(YOLO_MODEL_PATH)

def _get_yunet(width, height):
    detector = cv2.FaceDetectorYN.create(YUNET_MODEL_PATH, "", (width, height))
    detector.setScoreThreshold(0.5)
    return detector


def _iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0


def _detect(image_bytes: bytes):
    """Run YOLO + YuNet on raw image bytes. Return (head_count, annotated_jpeg_bytes)."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")

    h, w = img.shape[:2]
    annotated = img.copy()

    # ── YOLO (person class = 0) ──────────────────────────────────────
    results = _yolo.predict(img, conf=0.25, imgsz=640, verbose=False)
    yolo_boxes = []
    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == 0:  # person
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                yolo_boxes.append((x1, y1, x2, y2, conf))

    # ── YuNet (face detection) ───────────────────────────────────────
    yunet = _get_yunet(w, h)
    _, faces = yunet.detect(img)
    face_boxes = []
    if faces is not None:
        for face in faces:
            fx, fy, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
            fconf = float(face[-1])
            face_boxes.append((fx, fy, fx + fw, fy + fh, fconf))

    # ── Merge: drop face boxes that overlap a YOLO body box ──────────
    extra_faces = []
    for fb in face_boxes:
        covered = any(_iou(fb[:4], yb[:4]) > 0.3 for yb in yolo_boxes)
        if not covered:
            extra_faces.append(fb)

    all_detections = yolo_boxes + extra_faces
    head_count = len(all_detections)

    # ── Draw annotations ─────────────────────────────────────────────
    for i, (x1, y1, x2, y2, conf) in enumerate(yolo_boxes):
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Person {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 8, y1), (0, 255, 0), -1)
        cv2.putText(annotated, label, (x1 + 4, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    for (x1, y1, x2, y2, conf) in extra_faces:
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
        label = f"Face {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 8, y1), (255, 0, 0), -1)
        cv2.putText(annotated, label, (x1 + 4, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # ── People count banner ──────────────────────────────────────────
    banner = f"People Count: {head_count}"
    (tw, th), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    cv2.rectangle(annotated, (10, 10), (20 + tw + 10, 20 + th + 10), (0, 0, 0), -1)
    cv2.putText(annotated, banner, (20, 20 + th),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

    # ── Encode to JPEG bytes ─────────────────────────────────────────
    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return head_count, buf.tobytes()


# ── API Endpoint ─────────────────────────────────────────────────────

@app.post("/detect")
async def detect(image: UploadFile = File(...)):
    contents = await image.read()
    try:
        head_count, annotated_bytes = _detect(contents)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    encoded = base64.b64encode(annotated_bytes).decode("utf-8")
    return {"head_count": head_count, "annotated_image": encoded}


@app.get("/")
def root():
    return {"status": "ok", "message": "Crowd Detection Model API. POST an image to /detect"}
```

#### 3b. `requirements.txt` — Model API dependencies

```
fastapi==0.115.0
uvicorn[standard]==0.30.0
python-multipart==0.0.9
ultralytics==8.2.0
opencv-python-headless==4.10.0.84
numpy==1.26.4
torch==2.2.2
torchvision==0.17.2
```

#### 3c. `Dockerfile` — Container definition for HF Spaces

```dockerfile
FROM python:3.11-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model files
COPY . .

# HF Spaces expects port 7860
EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
```

#### 3d. `README.md` — Space metadata (required by HF)

```yaml
---
title: Crowd Detection Model
emoji: 👥
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---
```

### Step 4: Add Your Model Files

Copy your model weights into the Space folder:

```bash
# From your main project directory
cp models/yolov8m.pt  ../crowd-detection-model/
cp models/face_detection_yunet_2023mar.onnx  ../crowd-detection-model/
```

Your Space folder should now look like:

```
crowd-detection-model/
├── app.py
├── Dockerfile
├── README.md
├── requirements.txt
├── yolov8m.pt
└── face_detection_yunet_2023mar.onnx
```

### Step 5: Push to Hugging Face

```bash
cd crowd-detection-model

# If not logged in, authenticate first:
# pip install huggingface_hub
# huggingface-cli login
# (paste your HF access token when prompted)

git add .
git commit -m "Initial model API deployment"
git push
```

### Step 6: Wait for the Build

1. Go to your Space page: `https://huggingface.co/spaces/<your-username>/crowd-detection-model`
2. Click the **"Building"** status to watch logs
3. First build downloads PyTorch (~2GB) — this can take **5–10 minutes**
4. Once running, you'll see: `{"status": "ok", "message": "Crowd Detection Model API..."}`

### Step 7: Test the Model API

```bash
# Test with curl
curl -X POST "https://<your-username>-crowd-detection-model.hf.space/detect" \
     -F "image=@test_image.jpg"
```

Expected response:
```json
{
  "head_count": 15,
  "annotated_image": "/9j/4AAQSkZJRg..."
}
```

### Step 8: Connect the Web App

Set the `MODEL_API_URL` environment variable in your Flask web app to point to the Space:

```
MODEL_API_URL=https://<your-username>-crowd-detection-model.hf.space/detect
```

Your `detection.py` already uses this variable — the web app will now send images to your HF Space.

### Hugging Face Spaces — Important Notes

| Topic | Detail |
|---|---|
| **Free tier** | 2 vCPU, 16GB RAM — sufficient for YOLOv8m inference |
| **Cold starts** | Free Spaces sleep after ~48 hours of inactivity. First request after sleep takes 1–3 minutes (container rebuild). Upgrade to a persistent Space ($0/month with HF Pro) to avoid this |
| **Model file size** | Git LFS is used automatically for files >10MB. HF handles this — just `git push` |
| **Timeout** | Default request timeout is 60s. For large images, processing stays well within this |
| **Logs** | View live logs on your Space page → **Logs** tab |
| **Secrets** | If you need env vars in the model Space, add them in Space **Settings → Variables and secrets** |
| **Custom domain** | Available for paid Spaces |

### Troubleshooting HF Spaces

| Issue | Solution |
|---|---|
| Build fails with OOM | PyTorch is large. Use `python:3.11-slim` base and `--no-cache-dir` |
| `No module named 'cv2'` | Ensure `libgl1-mesa-glx` and `libglib2.0-0` are in the Dockerfile |
| Space won't start | Check the **Logs** tab. Common: wrong port (must be 7860), missing model file |
| Model returns 500 | Test locally first with `uvicorn app:app --port 7860`, send an image to `/detect` |
| Slow first request | Cold start. The model loads into memory on the first request after Space wakes |
| Git push rejected (LFS) | Run `git lfs install` then `git lfs track "*.pt" "*.onnx"` before committing |

---

## Local Development Setup

### Prerequisites

- Python 3.11+
- A [MongoDB Atlas](https://www.mongodb.com/atlas) free cluster
- A [Cloudinary](https://cloudinary.com) free account
- A running Model API (see [Deploying the Model API on Hugging Face Spaces](#deploying-the-model-api-on-hugging-face-spaces))

### 1. Clone and create virtual environment

```bash
git clone https://github.com/mourishantony/crowd_detection2.git
cd crowd_detection2
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set environment variables

Create a `.env` file in the project root (never commit this):

```env
MONGODB_URI=mongodb+srv://<user>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_api_key
CLOUDINARY_API_SECRET=your_api_secret
MODEL_API_URL=https://<your-username>-crowd-detection-model.hf.space/detect
```

### 4. Run the app

```bash
python app.py
```

Visit `http://localhost:5000`

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `MONGODB_URI` | Yes | MongoDB Atlas connection string (SRV format) |
| `CLOUDINARY_CLOUD_NAME` | Yes | Cloudinary cloud name |
| `CLOUDINARY_API_KEY` | Yes | Cloudinary API key |
| `CLOUDINARY_API_SECRET` | Yes | Cloudinary API secret |
| `MODEL_API_URL` | Yes | Full URL to the HF Spaces model API `/detect` endpoint |

---

## Deploying the Web App to Render

### 1. Push your code to GitHub

```bash
git add .
git commit -m "your message"
git push
```

### 2. Create a Render Web Service

1. Go to [render.com](https://render.com) → **New → Web Service**
2. Connect your GitHub repository
3. Render auto-detects the `Dockerfile`

### 3. Add environment variables on Render

In your service → **Environment** tab, add all five variables from the table above.

### 4. Deploy

Render will build the Docker image and deploy automatically on every push to `main`.

The `Dockerfile` runs:
```
gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 300
```

- `--workers 1` — keeps memory usage low on free tier
- `--timeout 300` — allows enough time for the model API to process (including HF Space cold starts)

---

## Admin Panel

Access at `/admin` with the credentials configured in `config.py`.

| Page | URL | Description |
|---|---|---|
| Login | `/admin` | Admin authentication |
| Dashboard | `/admin/dashboard` | Analytics, recent uploads, daily chart |
| Photos | `/admin/photos` | Gallery with lightbox, filter by place, delete |
| Places | `/admin/places` | Add, rename, delete places/events |

Default credentials (change in `config.py` before deploying):
```
Username: kgadmin
Password: kgadmin@2026
```

---

## API Reference

### `POST /upload` (Web App)

Upload an image for crowd detection.

**Form fields:**

| Field | Type | Description |
|---|---|---|
| `place` | string | Event/place name (new places are created automatically) |
| `image` | file | Image file (JPG, PNG, BMP, WEBP, TIFF) |

**Success response:**
```json
{
  "success": true,
  "head_count": 42,
  "event": "Ground"
}
```

**Error response:**
```json
{
  "success": false,
  "error": "Error message"
}
```

### `POST /detect` (Model API on HF Spaces)

Send an image to the model API for detection.

**Form field:**

| Field | Type | Description |
|---|---|---|
| `image` | file | Image file (any common format) |

**Response:**
```json
{
  "head_count": 15,
  "annotated_image": "<base64-encoded JPEG>"
}
```

---

## Notes

- The **web app does NOT run any ML models locally** — all detection is done by the remote HF Spaces model API
- The original uploaded image is **never stored permanently** — it is deleted from disk immediately after being sent to the model API
- Only the annotated image (with bounding boxes drawn) is stored, on Cloudinary
- MongoDB stores: event name, Cloudinary URL, Cloudinary public ID (for deletion), head count, and timestamp
- The `uploads/` folder is used only as a temporary scratch space
- If the HF Space is sleeping (free tier), the first request after inactivity may take 1–3 minutes while the container restarts
