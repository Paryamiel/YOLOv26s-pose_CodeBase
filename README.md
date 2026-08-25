# YOLOv26s-pose CodeBase

A GPU-accelerated, real-time human pose-estimation pipeline built on the Ultralytics YOLO framework and deployed with NVIDIA TensorRT engines. The repository ships five PyTorch weights (YOLO26n / YOLO26s / YOLO26m pose, YOLOv8s-pose, YOLO11n-pose), a ready-to-run TensorRT engine (`yolo26s-pose.engine`), a one-shot export script that converts any `.pt` weight into an optimized `.engine`, and a live webcam inference demo that runs pose estimation at 1920×1080 with an on-screen FPS counter.

The pipeline is intended for NVIDIA-CUDA-equipped workstations where latency matters — typical use cases include sports analytics, fitness coaching, sign-language recognition, action recognition preprocessing, and any robotic / vision stack that needs 17-keypoint COCO-format body landmarks at 30+ FPS.

---

## Table of Contents

1. [Key Features](#key-features)
2. [Repository Structure](#repository-structure)
3. [System Requirements](#system-requirements)
4. [Software Stack](#software-stack)
5. [Installation](#installation)
6. [Models Bundled in This Repo](#models-bundled-in-this-repo)
7. [Exporting a PyTorch Model to TensorRT](#exporting-a-pytorch-model-to-tensorrt)
8. [Running Real-Time Pose Estimation](#running-real-time-pose-estimation)
9. [Configuration Reference](#configuration-reference)
10. [Understanding the Output](#understanding-the-output)
11. [Performance Optimization Guide](#performance-optimization-guide)
12. [Benchmark Expectations](#benchmark-expectations)
13. [Troubleshooting](#troubleshooting)
14. [License & Acknowledgements](#license--acknowledgements)

---

## Key Features

- **TensorRT-accelerated inference** — Pre-built `yolo26s-pose.engine` runs at 1920×1080 with sub-frame latency on RTX-class GPUs.
- **Multi-model support** — Ships nano / small / medium YOLO26-pose variants plus YOLOv8s-pose and YOLO11n-pose for cross-architecture comparison.
- **One-shot export script** — `export_tensorrt.py` converts any Ultralytics `.pt` weight to a TensorRT engine with configurable batch size and dynamic batching.
- **Live webcam pipeline** — `pose_test.py` captures 1920×1080 frames from webcam index 0, runs TensorRT inference, draws 17 COCO keypoints + skeleton, and overlays a real-time FPS counter.
- **Standard COCO keypoint format** — Output produces `(x, y, visibility)` triples for the 17 standard COCO body landmarks, compatible with downstream action-recognition and biomechanics tooling.

---

## Repository Structure

```
YOLOv26s-pose_CodeBase/
├── README.md                  # This document
├── pose_test.py               # Real-time webcam pose-estimation demo (TensorRT engine)
├── export_tensorrt.py         # .pt → .engine conversion script (configurable batch)
├── yolo26n-pose.pt            # YOLO26 nano-pose PyTorch weights     (~7.6 MB)
├── yolo26s-pose.pt            # YOLO26 small-pose PyTorch weights    (~24 MB)
├── yolo26m-pose.pt            # YOLO26 medium-pose PyTorch weights   (~47 MB)
├── yolo26s-pose.engine        # Pre-built TensorRT engine            (~47 MB)
├── yolov8s-pose.pt            # YOLOv8 small-pose PyTorch weights    (~23 MB)
├── yolo11n-pose.pt            # YOLO11 nano-pose PyTorch weights     (~6.0 MB)
├── bus.jpg                    # Sample test image
└── __pycache__/               # Python bytecode cache (auto-generated)
```

### Script responsibilities

| File | Purpose | Input | Output |
|---|---|---|---|
| `pose_test.py` | Live webcam pose estimation | `yolo26s-pose.engine` + webcam index 0 | Annotated 1920×1080 window with FPS overlay |
| `export_tensorrt.py` | Convert PyTorch weights to TensorRT | Any `.pt` weight (default `yolo26s-pose.pt`) | New `.engine` file written next to the source |

---

## System Requirements

### Hardware

| Component | Minimum | Recommended | Notes |
|---|---|---|---|
| **GPU** | NVIDIA GTX 1660 (6 GB) | NVIDIA RTX 3060 / 4060 (8–12 GB) or better | **Must be NVIDIA.** AMD ROCm and Intel Arc are not supported by the TensorRT engine path. |
| **GPU Compute Capability** | 7.0 (Volta) | 8.6+ (Ampere / Ada) | TensorRT 8.6+ requires CC ≥ 7.0. TensorRT 10 requires CC ≥ 7.5. |
| **GPU VRAM** | 4 GB free | 8 GB free | The `yolo26s-pose.engine` needs ~2 GB at inference; export step needs ~6 GB. |
| **CPU** | 4 cores, 2.5 GHz | 8 cores, 3.0 GHz+ | For frame pre/post-processing parallelism. |
| **System RAM** | 8 GB | 16 GB+ | TensorRT engine build step peaks at ~4 GB RAM. |
| **Webcam** | Any USB camera supporting 1080p | Logitech C920/Brio, or RTSP IP camera via `cv2.VideoCapture("rtsp://...")` | `pose_test.py` opens index 0 at 1920×1080. |
| **Storage** | 1 GB free | 5 GB free (for engine cache, source weights, swap) | Engines are ~47 MB each; multiple variants add up. |

### Operating System

| OS | Status | Notes |
|---|---|---|
| **Ubuntu 22.04 LTS** | ✅ Primary target | Tested reference platform. |
| **Ubuntu 24.04 LTS** | ✅ Supported | Requires CUDA 12.4+ / cuDNN 9.x. |
| **Windows 11** | ✅ Supported | Install CUDA Toolkit + cuDNN separately; use Anaconda to avoid DLL hell. |
| **Windows 10** | ⚠️ Best-effort | TensorRT 10 may require newer drivers. |
| **macOS (Apple Silicon)** | ❌ Unsupported | TensorRT does not exist for macOS. Use `.pt` directly on Metal via `device=mps` — slower. |
| **WSL2 (Ubuntu on Windows)** | ⚠️ Experimental | TensorRT engine built inside WSL2 cannot be loaded on Windows host and vice-versa (engine files are GPU+driver+OS specific). |

### NVIDIA Driver

| TensorRT Version | Min Driver (Linux) | Min Driver (Windows) |
|---|---|---|
| TensorRT 10.x | ≥ 545.23.06 | ≥ 545.84 |
| TensorRT 8.6.x | ≥ 535.x | ≥ 535.x |

Check with `nvidia-smi` — the right-hand column of the top banner reports the maximum CUDA version supported by the installed driver.

---

## Software Stack

The pipeline is built on top of the Ultralytics ecosystem. The tested reference versions are:

| Layer | Package | Tested Version | Purpose |
|---|---|---|---|
| Deep-learning framework | `torch` | 2.4.0 (CUDA 12.1 build) | Tensor operations, autograd |
| CUDA backend | `torch` wheel | `cu121` | CUDA runtime bundled in wheel |
| cuDNN | (bundled in `torch`) | 8.9.x | Deep-learning primitives |
| Inference SDK | `ultralytics` | 8.3.x | YOLO model loading + `model.export(format="engine")` |
| Inference accelerator | TensorRT | 10.x (8.6+ acceptable) | `.engine` building and execution |
| Computer vision | `opencv-python` | 4.10.x | Webcam capture, frame drawing, display |
| Image / array | `numpy` | 1.26.x | Frame tensor manipulation |
| Python | CPython | 3.10 – 3.12 | Python runtime (**3.13 may work but is not yet officially supported by Ultralytics**) |

> **Note on Python 3.13**: The `__pycache__/export_tensorrt.cpython-313.pyc` and `pose_test.cpython-313.pyc` files in this repo indicate the author ran Python 3.13. It works for inference but some downstream packages (e.g. specific `onnx` / `onnx-graphsurgeon` builds) may not yet ship 3.13 wheels. For maximum stability, prefer Python 3.10–3.12.

---

## Installation

The instructions below assume Ubuntu 22.04 LTS. Windows users should run the equivalent commands inside Anaconda Prompt and skip step 1 (NVIDIA driver installation is GUI-based on Windows).

### Step 1 — Install the NVIDIA driver (Linux only)

```bash
sudo apt update
sudo apt install -y nvidia-driver-535
sudo reboot
# After reboot:
nvidia-smi   # Should print your GPU + driver version
```

### Step 2 — Create a Python environment

```bash
# Option A: venv (built-in)
python3 -m venv yolo_pose_env
source yolo_pose_env/bin/activate
python -m pip install --upgrade pip wheel

# Option B: conda (recommended if you have Anaconda/Miniconda)
conda create -n yolo_pose python=3.11 -y
conda activate yolo_pose
```

### Step 3 — Install PyTorch with CUDA support

The CUDA version chosen here must be ≤ the value reported by `nvidia-smi`'s "CUDA Version" banner.

```bash
# CUDA 12.1 build (most common)
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8 build (fallback for older drivers)
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu118
```

Verify GPU detection:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### Step 4 — Install Ultralytics and OpenCV

```bash
pip install ultralytics==8.3.0 opencv-python==4.10.0.84 numpy==1.26.4
```

### Step 5 — Install TensorRT

You have two options.

**Option A — pip wheel (simplest, recommended)**:

```bash
pip install tensorrt
```

**Option B — NVIDIA Tarball (more reliable for older CUDA)**:

1. Download the appropriate TensorRT tarball from https://developer.nvidia.com/tensorrt/download/8x (account required).
2. Extract and install Python bindings:

```bash
# Example for TensorRT 10.3 + Python 3.11
tar -xzvf TensorRT-10.3.0.26.Linux.x86_64-gnu.cuda-12.4.tar.gz
cd TensorRT-10.3.0.26/python
pip install tensorrt-10.3.0-cp311-none-linux_x86_64.whl
cd ../uff
pip install uff-0.6.9-py2.py3-none-any.whl
```

3. Add TensorRT `lib/` to `LD_LIBRARY_PATH`:

```bash
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/TensorRT-10.3.0.26/lib' >> ~/.bashrc
source ~/.bashrc
```

Verify the install:

```bash
python -c "import tensorrt; print('TensorRT version:', tensorrt.__version__)"
```

### Step 6 — Install ONNX + graphsurgeon (required only for the export step)

```bash
pip install onnx==1.16.1 onnx-graphsurgeon --extra-index-url https://pypi.nvidia.com
```

### Step 7 — Verify the full stack

```bash
python - <<'PY'
import torch, ultralytics, cv2, tensorrt, onnx
print('torch         :', torch.__version__, '| CUDA:', torch.cuda.is_available())
print('ultralytics   :', ultralytics.__version__)
print('opencv        :', cv2.__version__)
print('tensorrt      :', tensorrt.__version__)
print('onnx          :', onnx.__version__)
if torch.cuda.is_available():
    print('GPU            :', torch.cuda.get_device_name(0))
PY
```

All lines should print version numbers without raising any `ImportError`.

---

## Models Bundled in This Repo

The repository ships five Ultralytics pose models. All of them output the standard **17-keypoint COCO format** (`nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle`).

| File | Family | Size | Params | Use case |
|---|---|---|---|---|
| `yolo26n-pose.pt`  | YOLO26 nano     | 7.6 MB | ~2 M   | Edge devices, embedded, low-power. |
| `yolo26s-pose.pt`  | YOLO26 small    | 24 MB  | ~9 M   | **Default for this repo** — best speed/accuracy trade-off. |
| `yolo26m-pose.pt`  | YOLO26 medium   | 47 MB  | ~26 M  | High-accuracy applications; slower inference. |
| `yolov8s-pose.pt`  | YOLOv8 small    | 23 MB  | ~11 M  | Legacy comparison baseline. |
| `yolo11n-pose.pt`  | YOLO11 nano     | 6.0 MB | ~2 M   | Latest Ultralytics release; tiny. |

### Already-built engine

`yolo26s-pose.engine` is a pre-built TensorRT engine compiled from `yolo26s-pose.pt` with `BATCH_SIZE=16` and `DYNAMIC_BATCHING=True`. **Engine files are tied to the specific GPU, driver, and TensorRT version they were built on** — if you change any of these, you must re-export the engine using `export_tensorrt.py`.

---

## Exporting a PyTorch Model to TensorRT

The `export_tensorrt.py` script converts a `.pt` weight into an optimized `.engine` file. Edit the configuration block at the top of the file:

```python
MODEL_PATH = "yolo26s-pose.pt"   # Source PyTorch weights file (.pt)
BATCH_SIZE = 16                    # Maximum batch size (e.g. 1, 4, 8, 16, 32)
DYNAMIC_BATCHING = True            # Allow flexible batch sizes from 1 up to BATCH_SIZE
```

### What each parameter does

| Parameter | Effect |
|---|---|
| `MODEL_PATH` | Any Ultralytics-compatible `.pt` pose weight. You can swap `yolo26s-pose.pt` for `yolo26n-pose.pt`, `yolo26m-pose.pt`, `yolov8s-pose.pt`, or `yolo11n-pose.pt` to export a different architecture. |
| `BATCH_SIZE` | The maximum number of frames the engine can process in a single forward pass. Larger batches improve throughput when feeding many frames but increase VRAM usage linearly. |
| `DYNAMIC_BATCHING` | When `True`, the engine accepts any batch size from 1 up to `BATCH_SIZE`. When `False`, the engine accepts exactly `BATCH_SIZE` frames per call — slightly faster per call but inflexible. |

### Running the export

```bash
# From the project root
python export_tensorrt.py
```

Expected console output:

```
CUDA Available: True
Target GPU: NVIDIA GeForce RTX 4060
Loading PyTorch model 'yolo26s-pose.pt'...
Exporting model to TensorRT (.engine format)...
 -> Batch Size: 16
 -> Dynamic Batching: True
Export completed successfully! Engine saved at: yolo26s-pose.engine
```

The script typically takes **3–10 minutes** on an RTX 3060/4060 depending on the model size. During export, TensorRT performs kernel auto-tuning, which is the slow part.

### Exporting a different model

Edit `MODEL_PATH` (or pass an environment override) and re-run:

```bash
# One-shot export of the nano model
python -c "
import export_tensorrt as e
e.MODEL_PATH = 'yolo26n-pose.pt'
e.BATCH_SIZE = 8
e.export_to_tensorrt()
"
```

---

## Running Real-Time Pose Estimation

`pose_test.py` is a self-contained demo. It loads the pre-built `yolo26s-pose.engine`, opens webcam index 0 at 1920×1080, and runs pose estimation on every frame in a tight loop.

### Run the demo

```bash
python pose_test.py
```

You should see a window titled **"YOLO Pose Detection (TensorRT GPU)"** showing your webcam feed with:
- Skeleton lines connecting the 17 COCO keypoints.
- A keypoint dot at each detected landmark.
- A bounding box around each detected person.
- A green **"FPS (TensorRT GPU): XX.X"** counter in the top-left corner.

### Controls

| Key | Action |
|---|---|
| `q` | Quit the demo. |
| `ESC` | Quit the demo. |

### Switching to a different engine

By default `pose_test.py` loads `yolo26s-pose.engine`. To use a different engine, edit line 6 of the file:

```python
model = YOLO("yolo26n-pose.engine")   # Use the nano model instead
```

### Using a video file or RTSP stream instead of the webcam

Replace line 9:

```python
# Original (webcam index 0)
cap = cv2.VideoCapture(0)

# Video file
cap = cv2.VideoCapture("path/to/video.mp4")

# RTSP IP camera
cap = cv2.VideoCapture("rtsp://user:pass@192.168.1.10:554/stream")
```

---

## Configuration Reference

### `export_tensorrt.py` — full parameter reference

The Ultralytics `model.export()` call underneath accepts many parameters beyond the three surfaced in this script. To customize further, edit the `model.export(...)` block in `export_to_tensorrt.py`:

```python
exported_path = model.export(
    format="engine",        # Output format. Other options: "onnx", "openvino", "coreml", "tflite"
    device=0,               # GPU index (0 = primary GPU)
    batch=BATCH_SIZE,       # Maximum batch size
    dynamic=DYNAMIC_BATCHING,  # Allow variable batch from 1..batch
    half=True,              # FP16 quantization — ~2× faster, minimal accuracy loss. Default ON in Ultralytics.
    int8=False,             # INT8 quantization — requires calibration data, fastest but lossy.
    workspace=4,            # TensorRT workspace size in GB (default 4 GB). Increase if you hit "out of memory".
    simplify=True,          # Simplify the ONNX graph before building the engine.
    opset=12,               # ONNX opset version. 12 is safest for TensorRT 10.x.
    imgsz=640               # Input image size. YOLO26-pose default is 640×640.
)
```

### `pose_test.py` — runtime tunables

| Line | Variable | Default | Tunable effect |
|---|---|---|---|
| 6 | `YOLO("yolo26s-pose.engine")` | `yolo26s-pose.engine` | Path to TensorRT engine. |
| 9 | `cv2.VideoCapture(0)` | `0` | Camera index or file/RTSP URL. |
| 10 | `CAP_PROP_FRAME_WIDTH` | `1920` | Capture width. Lower to 1280 for low-end GPUs. |
| 11 | `CAP_PROP_FRAME_HEIGHT` | `1080` | Capture height. Lower to 720 for low-end GPUs. |
| 29 | `model(frame, verbose=False)` | `verbose=False` | Set to `True` to print per-frame inference metadata. |
| 47 | `FONT_HERSHEY_SIMPLEX` | — | Swap to `FONT_HERSHEY_DUPLEX` for thicker FPS text. |

### Inference-time arguments you can add

Inside `pose_test.py`, the inference line can be expanded:

```python
results = model(
    frame,
    verbose=False,
    conf=0.25,       # Detection confidence threshold. Lower = more detections but more false positives.
    iou=0.7,         # Non-Maximum Suppression IoU threshold. Lower = fewer overlapping boxes.
    device=0,        # Force GPU 0.
    imgsz=640        # Inference image size. Larger = more accurate but slower.
)
```

---

## Understanding the Output

Inside the inference loop, `pose_test.py` extracts three representations of the keypoints:

```python
for result in results:
    keypoints = result.keypoints
    if keypoints is not None:
        xy   = keypoints.xy      # (N, 17, 2)  — pixel coordinates (x, y)
        xyn  = keypoints.xyn     # (N, 17, 2)  — normalized [0..1] coordinates
        kpts = keypoints.data    # (N, 17, 3)  — (x, y, visibility_score)
```

Where:
- `N` = number of detected persons in the frame.
- `17` = the COCO keypoint count (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles).
- `visibility_score` ranges from 0.0 (not visible / occluded) to 1.0 (clearly visible).

### Drawing the annotated frame

`results[0].plot()` (line 40) renders the skeleton, keypoint dots, and bounding boxes directly onto the source frame and returns the modified NumPy array. This is what gets displayed in the OpenCV window.

---

## Performance Optimization Guide

The pipeline is already highly optimized (TensorRT FP16 engine), but the following levers can push it further:

### 1. Choose the right model size

| Target FPS @ 1080p | Recommended model | Recommended batch |
|---|---|---|
| 60+ FPS | `yolo26n-pose` | 1 (live stream) |
| 30–50 FPS | `yolo26s-pose` (this repo's default) | 1 (live stream) |
| 15–25 FPS | `yolo26m-pose` | 1 (live stream) |
| Max throughput (batched) | `yolo26n-pose` or `yolo26s-pose` | 8–16 (offline processing) |

### 2. Use FP16 (already on by default)

The Ultralytics `export(format="engine")` call uses FP16 by default. Verify with:

```bash
python -c "from ultralytics import YOLO; m=YOLO('yolo26s-pose.engine'); print(m.task)"
```

### 3. Tune the TensorRT workspace

If the export step fails with `out of memory`, increase `workspace=8` (or `16`) in `model.export(...)`.

### 4. Lower the capture resolution

If your GPU cannot sustain 30 FPS at 1920×1080, drop to 1280×720 in `pose_test.py` (lines 10–11). This roughly halves inference time.

### 5. Skip every other frame

For very low-end GPUs, process every 2nd frame and reuse the previous frame's overlay:

```python
results = model(frame if frame_idx % 2 == 0 else prev_frame, verbose=False)
frame_idx += 1
```

### 6. Disable verbose OpenCV drawing

`results[0].plot()` draws all keypoints + skeletons + boxes. For maximum FPS, replace it with a custom minimal drawer that only plots visible keypoints (`visibility > 0.5`).

### 7. Batch processing for offline video

For offline video files, batch multiple frames:

```python
batch = [frame1, frame2, frame3, frame4]
results = model(batch, verbose=False)   # Processes 4 frames in one forward pass
```

This requires the engine to have been exported with `BATCH_SIZE ≥ 4`.

### 8. Avoid CPU↔GPU copies

Don't call `.cpu()` on result tensors unless you need to read them in Python. Each `.cpu()` call forces a synchronization that stalls the GPU pipeline.

### 9. Use `imgsz=480` for small objects

If your subjects are small in frame (e.g. wide-angle surveillance), increase `imgsz=960` for better detection — at the cost of inference latency. Conversely, for close-up subjects, `imgsz=480` is faster.

### 10. Pin CPU memory (advanced)

For batched pipelines, wrap input tensors with `torch.tensor(...).pin_memory()` before passing to the model. This enables asynchronous host→device copies.

---

## Benchmark Expectations

Approximate FPS on a single 1920×1080 webcam stream, batch=1, FP16:

| GPU | `yolo26n-pose.engine` | `yolo26s-pose.engine` | `yolo26m-pose.engine` |
|---|---|---|---|
| RTX 4090       | 180+ FPS | 130+ FPS | 75+ FPS |
| RTX 4060       | 90+ FPS  | 55+ FPS  | 30+ FPS  |
| RTX 3060       | 75+ FPS  | 45+ FPS  | 25+ FPS  |
| GTX 1660 Ti    | 45+ FPS  | 28+ FPS  | 15+ FPS  |
| GTX 1050 Ti    | 22+ FPS  | 12+ FPS  | 6+ FPS   |

These are rough estimates — actual throughput depends on driver version, CPU pre-processing overhead, and OpenCV drawing cost. To measure pure model latency (without display), comment out lines 40–55 in `pose_test.py` and re-run; the FPS counter will then reflect pure inference throughput.

---

## Troubleshooting

### `RuntimeError: TensorRT engine file is not compatible with this version of TensorRT`

The bundled `yolo26s-pose.engine` was built with a specific TensorRT version. Re-export the engine:

```bash
python export_tensorrt.py
```

This regenerates the engine with your locally installed TensorRT.

### `CUDA out of memory` during export

Reduce `BATCH_SIZE` in `export_tensorrt.py` to 4 or 1, or increase `workspace` (see [Configuration Reference](#configuration-reference)).

### `Unable to fetch camera frame` (printed in console by `pose_test.py`)

Causes:
1. **Wrong camera index** — try `cv2.VideoCapture(1)` or `cv2.VideoCapture(2)`.
2. **Camera is in use** — close Zoom / Teams / OBS / browser tabs that may have grabbed the camera.
3. **Camera does not support 1920×1080** — lower to `1280×720` (lines 10–11).
4. **Permission denied (Linux)** — `sudo usermod -aG video $USER`, then log out and back in.

### `cv2.imshow` window doesn't appear

On headless Linux servers there is no display server. Run the script on a machine with a desktop environment, or replace `cv2.imshow(...)` with `cv2.imwrite(...)` to save frames to disk.

### `ImportError: libnvinfer.so.8: cannot open shared object file`

TensorRT libraries aren't on your `LD_LIBRARY_PATH`. Fix:

```bash
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
# or, if using the Tarball install:
export LD_LIBRARY_PATH=/path/to/TensorRT/lib:$LD_LIBRARY_PATH
```

### TensorRT engine was built on a different GPU

TensorRT engines are **GPU-architecture-specific**. An engine built on an RTX 3060 will not load on an RTX 4060 (different compute capability). Always re-export after upgrading your GPU:

```bash
python export_tensorrt.py
```

### Slow FPS despite TensorRT

1. Verify the model is loaded as `.engine`, not `.pt` (line 6 of `pose_test.py`).
2. Run `nvidia-smi` while the script is running — GPU utilization should be > 70%.
3. Check `htop` — if a single CPU core is at 100%, the bottleneck is OpenCV drawing, not inference. Disable `results[0].plot()` to confirm.
4. Ensure your power supply is delivering enough wattage (laptops on battery throttle GPU heavily).

### OpenCV errors when reading from RTSP

```bash
# Install the full OpenCV build with FFmpeg support
pip uninstall opencv-python -y
pip install opencv-python-headless opencv-contrib-python
```

---

## License & Acknowledgements

- **Ultralytics YOLO** — Released under the AGPL-3.0 license. Commercial use requires an Ultralytics Enterprise license. See https://github.com/ultralytics/ultralytics/blob/main/LICENSE for details.
- **TensorRT** — NVIDIA proprietary, free for internal business use; production deployment requires an NVIDIA AI Enterprise license in some scenarios. See https://developer.nvidia.com/tensorrt.
- **OpenCV** — Apache 2.0.
- **COCO Keypoint Format** — The 17-keypoint body landmark standard defined by the Common Objects in Context dataset.

This repository bundles pre-trained weights (`*.pt`, `*.engine`) which are subject to their respective upstream licenses. The Python source code in this repository is provided as-is for educational and research purposes.

---

### Citation

If you use this pipeline in academic work, please cite the Ultralytics YOLO papers:

```bibtex
@software{yolo,
  title = {Ultralytics YOLO},
  author = {Glenn Jocher and Ayush Chaurasia and Jing Qiu},
  year = {2023},
  url = {https://github.com/ultralytics/ultralytics}
}
```

---

**Built with:** Python 3.10+ · PyTorch 2.4 (CUDA 12.1) · Ultralytics 8.3 · TensorRT 10.x · OpenCV 4.10
