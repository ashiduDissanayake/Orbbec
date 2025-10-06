from ultralytics import YOLO

print("📥 Downloading YOLOv8 nano model...")
model = YOLO('yolov8n.pt')

print("🔄 Exporting to ONNX format...")
model.export(format='onnx', simplify=True)

print("✅ Export complete! File: yolov8n.onnx")
