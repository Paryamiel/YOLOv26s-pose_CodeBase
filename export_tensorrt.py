import torch
from ultralytics import YOLO

# =========================================================
# CONFIGURATION - MANIPULATE BATCH SIZE & SETTINGS HERE
# =========================================================
MODEL_PATH = "yolo26s-pose.pt"  # Source PyTorch weights file (.pt)
BATCH_SIZE = 16                 # Maximum batch size (e.g. 1, 4, 8, 16, 32)
DYNAMIC_BATCHING = True         # Allow flexible batch sizes from 1 up to BATCH_SIZE

def export_to_tensorrt():
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Target GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Warning: CUDA is not active. Ensure PyTorch GPU with CUDA is installed.")

    print(f"Loading PyTorch model '{MODEL_PATH}'...")
    model = YOLO(MODEL_PATH)

    print(f"Exporting model to TensorRT (.engine format)...")
    print(f" -> Batch Size: {BATCH_SIZE}")
    print(f" -> Dynamic Batching: {DYNAMIC_BATCHING}")

    # Export model to TensorRT engine for NVIDIA GPU
    exported_path = model.export(
        format="engine",
        device=0,
        batch=BATCH_SIZE,
        dynamic=DYNAMIC_BATCHING
    )
    print(f"Export completed successfully! Engine saved at: {exported_path}")

if __name__ == "__main__":
    export_to_tensorrt()
