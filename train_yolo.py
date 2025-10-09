#!/usr/bin/env python3
"""
Train YOLOv8 on Red Ball Dataset
Includes built-in augmentation
"""

from ultralytics import YOLO
import torch

# Check for GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🖥️  Training on: {device}")

# Load pretrained YOLOv8 nano model
model = YOLO('yolov8n.pt')

# Train with augmentation
results = model.train(
    data='Red_Ball.v1i.yolov8/data.yaml',  # Dataset config file (fixed extension)
    epochs=100,                    # Train for 100 epochs
    imgsz=640,                     # Image size
    batch=16,                      # Batch size (adjust based on GPU)
    device=device,
    
    # # AUGMENTATION SETTINGS (Built into YOLO)
    # hsv_h=0.015,                   # Hue augmentation (±1.5%)
    # hsv_s=0.7,                     # Saturation augmentation (±70%)
    # hsv_v=0.4,                     # Brightness augmentation (±40%)
    # degrees=15.0,                  # Rotation (±15°)
    # translate=0.1,                 # Translation (±10%)
    # scale=0.2,                     # Scale (80-120%)
    # shear=0.0,                     # No shearing (ball stays round)
    # perspective=0.0,               # No perspective (ball stays round)
    # flipud=0.0,                    # No vertical flip
    # fliplr=0.5,                    # 50% horizontal flip
    # mosaic=1.0,                    # Mosaic augmentation
    # mixup=0.1,                     # 10% mixup
    # copy_paste=0.0,                # No copy-paste (single ball)
    
    # TRAINING SETTINGS
    patience=20,                   # Early stopping patience
    save=True,                     # Save checkpoints
    save_period=10,                # Save every 10 epochs
    cache=False,                   # Don't cache (saves RAM)
    workers=4,                     # Data loading threads
    project='runs/train',          # Output directory
    name='red_ball_v1',            # Experiment name
    exist_ok=True,
    
    # OPTIMIZER SETTINGS
    optimizer='Adam',              # Adam optimizer
    lr0=0.001,                     # Initial learning rate
    lrf=0.01,                      # Final learning rate factor
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    
    # VALIDATION
    val=True,
    plots=True,                    # Generate training plots
    verbose=True
)

print("\n✅ Training complete!")
print(f"📁 Model saved to: runs/train/red_ball_v1/weights/best.pt")