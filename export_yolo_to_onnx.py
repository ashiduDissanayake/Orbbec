#!/usr/bin/env python3
"""
Export trained YOLO model to ONNX format for C++ integration
"""

from ultralytics import YOLO
from pathlib import Path

# Paths
MODEL_PATH = './runs/train/red_ball_v1/weights/best.pt'
EXPORT_DIR = './OrbbecSDK/examples/cpp/Sample-ColorDepthFallback/'

print("🔄 Exporting YOLO model to ONNX format...\n")

# Check if model exists
if not Path(MODEL_PATH).exists():
    print(f"❌ Model not found at: {MODEL_PATH}")
    exit(1)

# Load model
model = YOLO(MODEL_PATH)

# Export to ONNX
print(f"📦 Loading model from: {MODEL_PATH}")
onnx_path = model.export(
    format='onnx',           # Export to ONNX format
    imgsz=640,               # Input image size
    optimize=True,           # Optimize for inference
    simplify=True,           # Simplify model graph
    opset=12,                # ONNX opset version (compatible with OpenCV DNN)
    dynamic=False            # Fixed input shape for faster inference
)

print(f"\n✅ Model exported successfully!")
print(f"📁 ONNX model location: {onnx_path}")
print(f"\n💡 You can now use this model with OpenCV DNN in C++")
print(f"   Model file: {Path(onnx_path).name}")
print(f"\n📋 Model specs:")
print(f"   - Input size: 640x640")
print(f"   - Format: ONNX (OpenCV compatible)")
print(f"   - Classes: 1 (Red Ball)")
print(f"   - Optimized: Yes")
