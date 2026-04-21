"""
MNIST Neural Network Explorer — FastAPI Backend
Serves real-time training progress, predictions, and system metrics via WebSockets.
"""

import asyncio
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional, Set

import psutil
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))
from model import MNISTNet

# ─── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).parent.parent
STATIC_DIR = BASE_DIR / "static"
MODEL_PATH = BASE_DIR / "mnist_model.pth"
DATA_DIR   = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# ─── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(title="MNIST Neural Network Explorer", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global State ─────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = MNISTNet().to(device)
model.eval()

training_state = {
    "is_training": False,
    "status": "idle",          # idle | training | stopped | completed | error
    "current_epoch": 0,
    "total_epochs": 0,
    "current_batch": 0,
    "total_batches": 0,
    "loss": None,
    "train_acc": None,
    "val_acc": None,
    "speed": 0,                # samples / second
    "elapsed": 0,
    "eta": 0,
    "lr": 0.001,
    "history": {
        "epochs": [],
        "loss": [],
        "train_acc": [],
        "val_acc": [],
    },
    "model_trained": False,
    "last_inference_ms": 0,
}

stop_training_flag = threading.Event()
training_lock      = threading.Lock()

# WebSocket client sets
train_ws_clients:   Set[WebSocket] = set()
metrics_ws_clients: Set[WebSocket] = set()

# Load saved model on startup
if MODEL_PATH.exists():
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        training_state["status"]        = "completed"
        training_state["model_trained"] = True
        print(f"✓ Loaded saved model from {MODEL_PATH}")
    except Exception as e:
        print(f"⚠ Could not load saved model: {e}")

# ─── WebSocket Helpers ────────────────────────────────────────────────────────

async def broadcast(ws_set: Set[WebSocket], data: dict):
    dead = set()
    for ws in ws_set:
        try:
            await ws.send_json(data)
        except Exception:
            dead.add(ws)
    ws_set -= dead

# ─── Training Thread ──────────────────────────────────────────────────────────

def run_training(epochs: int, lr: float, batch_size: int, reset: bool, loop: asyncio.AbstractEventLoop):
    global model

    def push(data: dict):
        asyncio.run_coroutine_threadsafe(broadcast(train_ws_clients, data), loop)

    # Optionally reset model weights
    if reset:
        model = MNISTNet().to(device)

    with training_lock:
        training_state.update({
            "is_training": True,
            "status": "training",
            "total_epochs": epochs,
            "current_epoch": 0,
            "lr": lr,
            "history": {"epochs": [], "loss": [], "train_acc": [], "val_acc": []},
        })

    stop_training_flag.clear()

    try:
        # ── Load MNIST ──────────────────────────────────────────────────────
        push({"type": "status", "message": "Downloading / loading MNIST dataset…"})

        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_ds = torchvision.datasets.MNIST(str(DATA_DIR), train=True,  download=True, transform=tfm)
        val_ds   = torchvision.datasets.MNIST(str(DATA_DIR), train=False, download=True, transform=tfm)

        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=False)
        val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

        total_batches = len(train_loader)
        training_state["total_batches"] = total_batches

        push({"type": "status", "message": f"Dataset ready. {len(train_ds):,} train / {len(val_ds):,} val samples."})

        # ── Optimizer & loss ────────────────────────────────────────────────
        optimizer  = optim.Adam(model.parameters(), lr=lr)
        scheduler  = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
        criterion  = nn.CrossEntropyLoss()

        model.train()
        global_start = time.time()

        for epoch in range(1, epochs + 1):
            if stop_training_flag.is_set():
                break

            training_state["current_epoch"] = epoch
            epoch_loss = 0.0
            correct    = 0
            total      = 0
            epoch_t0   = time.time()

            for batch_idx, (data, target) in enumerate(train_loader):
                if stop_training_flag.is_set():
                    break

                data, target = data.to(device), target.to(device)
                data = data.view(-1, 784)

                optimizer.zero_grad()
                output = model(data)
                loss   = criterion(output, target)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                pred    = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total   += target.size(0)

                elapsed  = time.time() - global_start
                done     = (epoch - 1) * total_batches + batch_idx + 1
                all_done = epochs * total_batches
                speed    = (done * batch_size) / elapsed if elapsed > 0 else 0
                eta      = ((all_done - done) * batch_size) / speed if speed > 0 else 0

                training_state.update({
                    "current_batch": batch_idx + 1,
                    "loss":    epoch_loss / (batch_idx + 1),
                    "train_acc": correct / total,
                    "speed":   int(speed),
                    "elapsed": round(elapsed, 1),
                    "eta":     round(eta, 1),
                    "lr":      optimizer.param_groups[0]["lr"],
                })

                if batch_idx % 15 == 0:
                    push({
                        "type":         "batch",
                        "epoch":         epoch,
                        "total_epochs":  epochs,
                        "batch":         batch_idx + 1,
                        "total_batches": total_batches,
                        "loss":          round(epoch_loss / (batch_idx + 1), 4),
                        "train_acc":     round(correct / total, 4),
                        "speed":         int(speed),
                        "elapsed":       round(elapsed, 1),
                        "eta":           round(eta, 1),
                        "lr":            round(optimizer.param_groups[0]["lr"], 6),
                    })

            if stop_training_flag.is_set():
                break

            # ── Validation pass ─────────────────────────────────────────────
            model.eval()
            val_correct = 0
            val_total   = 0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    data = data.view(-1, 784)
                    output = model(data)
                    pred   = output.argmax(dim=1)
                    val_correct += (pred == target).sum().item()
                    val_total   += target.size(0)

            val_acc    = val_correct / val_total
            avg_loss   = epoch_loss / total_batches
            train_acc  = correct / total
            epoch_time = time.time() - epoch_t0

            model.train()
            scheduler.step()

            training_state["history"]["epochs"].append(epoch)
            training_state["history"]["loss"].append(round(avg_loss, 4))
            training_state["history"]["train_acc"].append(round(train_acc, 4))
            training_state["history"]["val_acc"].append(round(val_acc, 4))
            training_state["val_acc"] = val_acc

            push({
                "type":        "epoch",
                "epoch":        epoch,
                "total_epochs": epochs,
                "loss":         round(avg_loss, 4),
                "train_acc":    round(train_acc, 4),
                "val_acc":      round(val_acc, 4),
                "epoch_time":   round(epoch_time, 1),
                "history":      training_state["history"],
            })

        # ── Save model ──────────────────────────────────────────────────────
        torch.save(model.state_dict(), str(MODEL_PATH))
        model.eval()

        final = "stopped" if stop_training_flag.is_set() else "completed"
        training_state.update({
            "is_training":   False,
            "status":        final,
            "model_trained": True,
        })
        push({
            "type":    "complete",
            "status":  final,
            "message": "Training complete! Model saved." if final == "completed" else "Training stopped. Model saved.",
            "history": training_state["history"],
        })

    except Exception as exc:
        training_state.update({"is_training": False, "status": "error"})
        push({"type": "error", "message": str(exc)})
        raise

# ─── Pydantic Models ──────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    pixels: List[float]   # 784 values in [0, 1]

class TrainRequest(BaseModel):
    epochs:      int   = 5
    lr:          float = 0.001
    batch_size:  int   = 64
    reset_model: bool  = False

# ─── REST Endpoints ───────────────────────────────────────────────────────────

@app.get("/api/status")
async def get_status():
    return {
        "status":        training_state["status"],
        "model_trained": training_state["model_trained"],
        "device":        str(device),
        "has_cuda":      torch.cuda.is_available(),
        "model_params":  model.count_parameters(),
        "layer_info":    model.get_layer_info(),
        "layer_sizes":   MNISTNet.LAYER_SIZES,
        "layer_names":   MNISTNet.LAYER_NAMES,
    }

@app.post("/api/predict")
async def predict(req: PredictRequest):
    if len(req.pixels) != 784:
        raise HTTPException(400, "Expected exactly 784 pixel values (28×28).")

    t0 = time.perf_counter()

    tensor = torch.FloatTensor(req.pixels).to(device)
    tensor = (tensor - 0.1307) / 0.3081   # MNIST normalization
    tensor = tensor.unsqueeze(0)

    activations = model.get_activations(tensor)

    with torch.no_grad():
        output = model(tensor)
        probs  = torch.softmax(output, dim=-1).squeeze().cpu().tolist()

    ms = round((time.perf_counter() - t0) * 1000, 2)
    training_state["last_inference_ms"] = ms

    prediction = int(probs.index(max(probs)))
    return {
        "prediction":    prediction,
        "confidence":    round(max(probs), 4),
        "probabilities": [round(p, 4) for p in probs],
        "activations":   activations,
        "inference_ms":  ms,
    }

@app.post("/api/train/start")
async def start_training(req: TrainRequest):
    if training_state["is_training"]:
        raise HTTPException(400, "Training is already running.")

    loop = asyncio.get_running_loop()
    t = threading.Thread(
        target=run_training,
        args=(req.epochs, req.lr, req.batch_size, req.reset_model, loop),
        daemon=True,
    )
    t.start()
    return {"message": "Training started.", "config": req.dict()}

@app.post("/api/train/stop")
async def stop_training_endpoint():
    if not training_state["is_training"]:
        raise HTTPException(400, "No training is in progress.")
    stop_training_flag.set()
    return {"message": "Stop signal sent."}

@app.get("/api/train/state")
async def get_training_state():
    return training_state

@app.get("/api/sample")
async def get_sample(digit: Optional[int] = None):
    """Return a random MNIST test image (requires dataset downloaded)."""
    try:
        tfm = transforms.Compose([transforms.ToTensor()])
        ds  = torchvision.datasets.MNIST(str(DATA_DIR), train=False, download=False, transform=tfm)
    except Exception:
        raise HTTPException(404, "Dataset not available. Train the model first to download it.")

    import random
    if digit is not None:
        candidates = [i for i, (_, lbl) in enumerate(ds) if lbl == digit]
        if not candidates:
            raise HTTPException(404, f"No sample found for digit {digit}.")
        idx = random.choice(candidates[:200])  # only search first 200 to stay fast
    else:
        idx = random.randint(0, len(ds) - 1)

    img, label = ds[idx]
    pixels = img.squeeze().flatten().tolist()   # raw 0-1 values
    return {"pixels": pixels, "label": int(label)}

# ─── WebSocket: Training Updates ──────────────────────────────────────────────

@app.websocket("/ws/train")
async def ws_train(ws: WebSocket):
    await ws.accept()
    train_ws_clients.add(ws)
    try:
        await ws.send_json({"type": "state", **training_state})
        while True:
            await ws.receive_text()   # keep-alive ping
    except WebSocketDisconnect:
        pass
    finally:
        train_ws_clients.discard(ws)

# ─── WebSocket: System Metrics ────────────────────────────────────────────────

@app.websocket("/ws/metrics")
async def ws_metrics(ws: WebSocket):
    await ws.accept()
    metrics_ws_clients.add(ws)
    proc = psutil.Process()
    try:
        while True:
            cpu  = psutil.cpu_percent(interval=None)
            ram  = psutil.virtual_memory()
            pmem = proc.memory_info().rss / (1024 ** 2)  # MB

            await ws.send_json({
                "cpu_percent":      cpu,
                "cpu_count":        psutil.cpu_count(logical=True),
                "ram_percent":      ram.percent,
                "ram_used_gb":      round(ram.used  / (1024 ** 3), 2),
                "ram_total_gb":     round(ram.total / (1024 ** 3), 2),
                "ram_avail_gb":     round(ram.available / (1024 ** 3), 2),
                "process_ram_mb":   round(pmem, 1),
                "training_speed":   training_state.get("speed", 0),
                "is_training":      training_state["is_training"],
                "device":           str(device),
                "last_inference_ms": training_state["last_inference_ms"],
            })
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass
    finally:
        metrics_ws_clients.discard(ws)

# ─── Static Files ─────────────────────────────────────────────────────────────

if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print(f"🧠 MNIST Explorer running on http://localhost:8000")
    print(f"   Device: {device}")
    print(f"   Model params: {model.count_parameters():,}")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
