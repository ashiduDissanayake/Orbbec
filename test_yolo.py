#!/usr/bin/env python3
"""
Test YOLOv8 Red Ball Model
Evaluates performance on test dataset and generates detailed metrics
"""

from ultralytics import YOLO
import torch
from pathlib import Path

# Paths
MODEL_PATH = './runs/train/red_ball_v1/weights/best.pt'
DATA_YAML = './OrbbecSDK/Red_Ball_Detection.v1-v1.yolov8/data.yaml'
TEST_DIR = './OrbbecSDK/Red_Ball_Detection.v1-v1.yolov8/test/images'

# Check for GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🖥️  Testing on: {device}\n")

# Check if model exists
if not Path(MODEL_PATH).exists():
    print(f"❌ Model not found at: {MODEL_PATH}")
    print("Please train the model first using train_yolo.py")
    exit(1)

print(f"📦 Loading trained model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

# ============================================================================
# 1. VALIDATION ON TEST SET
# ============================================================================
print("\n" + "="*70)
print("📊 RUNNING VALIDATION ON TEST SET")
print("="*70)

metrics = model.val(
    data=DATA_YAML,
    split='test',           # Use test split
    device=device,
    batch=8,
    imgsz=640,
    conf=0.25,              # Confidence threshold
    iou=0.7,                # IoU threshold for NMS
    plots=True,             # Generate plots
    save_json=True,         # Save results as JSON
    verbose=True
)

# ============================================================================
# 2. PRINT DETAILED METRICS
# ============================================================================
print("\n" + "="*70)
print("📈 FINAL MODEL PERFORMANCE METRICS")
print("="*70)

# Box metrics
box_metrics = metrics.box
print(f"\n📦 Bounding Box Metrics:")
print(f"  ├─ Precision (P):     {box_metrics.p[0]:.4f} ({box_metrics.p[0]*100:.2f}%)")
print(f"  ├─ Recall (R):        {box_metrics.r[0]:.4f} ({box_metrics.r[0]*100:.2f}%)")
print(f"  ├─ mAP@50:            {box_metrics.map50:.4f} ({box_metrics.map50*100:.2f}%)")
print(f"  ├─ mAP@50-95:         {box_metrics.map:.4f} ({box_metrics.map*100:.2f}%)")
print(f"  └─ F1 Score:          {2 * (box_metrics.p[0] * box_metrics.r[0]) / (box_metrics.p[0] + box_metrics.r[0]):.4f}")

# Speed metrics
print(f"\n⚡ Speed Metrics:")
print(f"  ├─ Preprocess:        {metrics.speed['preprocess']:.2f} ms/image")
print(f"  ├─ Inference:         {metrics.speed['inference']:.2f} ms/image")
print(f"  ├─ Postprocess:       {metrics.speed['postprocess']:.2f} ms/image")
total_time = sum(metrics.speed.values())
print(f"  └─ Total:             {total_time:.2f} ms/image ({1000/total_time:.1f} FPS)")

# ============================================================================
# 3. QUALITY ASSESSMENT
# ============================================================================
print("\n" + "="*70)
print("✅ MODEL QUALITY ASSESSMENT")
print("="*70)

precision = box_metrics.p[0]
recall = box_metrics.r[0]
map50 = box_metrics.map50
map50_95 = box_metrics.map

# Grade the model
def grade_metric(value, excellent=0.95, good=0.85, fair=0.70):
    if value >= excellent:
        return "🟢 EXCELLENT"
    elif value >= good:
        return "🟡 GOOD"
    elif value >= fair:
        return "🟠 FAIR"
    else:
        return "🔴 NEEDS IMPROVEMENT"

print(f"\n📊 Performance Grades:")
print(f"  ├─ Precision:         {grade_metric(precision)}")
print(f"  ├─ Recall:            {grade_metric(recall)}")
print(f"  ├─ mAP@50:            {grade_metric(map50)}")
print(f"  └─ mAP@50-95:         {grade_metric(map50_95, 0.75, 0.60, 0.45)}")

# Overall assessment
overall_score = (precision + recall + map50 + map50_95) / 4
print(f"\n🎯 Overall Score:      {overall_score:.4f} ({overall_score*100:.2f}%)")

if overall_score >= 0.90:
    print(f"   Status:             🏆 PRODUCTION READY - Excellent performance!")
elif overall_score >= 0.80:
    print(f"   Status:             ✅ GOOD - Ready for deployment with monitoring")
elif overall_score >= 0.70:
    print(f"   Status:             ⚠️  ACCEPTABLE - May need more training or data")
else:
    print(f"   Status:             ❌ NEEDS WORK - Consider retraining with more data")

# ============================================================================
# 4. CONFUSION MATRIX & PLOTS
# ============================================================================
print(f"\n📁 Results saved to: runs/detect/val/")
print(f"   ├─ confusion_matrix.png    - Visual confusion matrix")
print(f"   ├─ F1_curve.png            - F1 score vs confidence")
print(f"   ├─ P_curve.png             - Precision curve")
print(f"   ├─ R_curve.png             - Recall curve")
print(f"   ├─ PR_curve.png            - Precision-Recall curve")
print(f"   └─ predictions.json        - Detailed predictions")

# ============================================================================
# 5. RECOMMENDATIONS
# ============================================================================
print("\n" + "="*70)
print("💡 RECOMMENDATIONS")
print("="*70)

if precision < 0.85:
    print("\n⚠️  Low Precision Detected:")
    print("   → Model is producing false positives (detecting balls where there aren't any)")
    print("   → Solutions: Increase confidence threshold, add more negative samples")

if recall < 0.85:
    print("\n⚠️  Low Recall Detected:")
    print("   → Model is missing some balls")
    print("   → Solutions: Lower confidence threshold, add more training examples")

if map50_95 < 0.60:
    print("\n⚠️  Low mAP@50-95:")
    print("   → Bounding boxes are not precise enough")
    print("   → Solutions: Train longer, use larger model (yolov8s, yolov8m)")

if overall_score >= 0.90:
    print("\n✨ Your model is performing exceptionally well!")
    print("   Next steps:")
    print("   1. Test on real-world camera feed")
    print("   2. Integrate with C++ ball tracking system")
    print("   3. Deploy and monitor performance")

print("\n" + "="*70)
print("✅ Testing complete!")
print("="*70)
print(f"\n📦 Best model ready at: {MODEL_PATH}")
print(f"   Use this model for inference and deployment.\n")
