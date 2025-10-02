#!/bin/bash
# Safe cleanup script - removes only unnecessary files
# Keeps essential SDK files and your custom code

echo "🧹 Cleaning up Orbbec Ball Tracking Project..."
echo ""

cd /Users/ashidudissanayake/Dev/Orbbec

# ============================================================================
# 1. Remove Orbbec SDK documentation (can reference online)
# ============================================================================
if [ -d "OrbbecSDK/doc" ]; then
    echo "📄 Removing SDK documentation..."
    rm -rf OrbbecSDK/doc/
fi

# ============================================================================
# 2. Remove misc files (drivers, config samples)
# ============================================================================
if [ -d "OrbbecSDK/misc" ]; then
    echo "📦 Removing misc files..."
    rm -rf OrbbecSDK/misc/
fi

# ============================================================================
# 3. Remove build artifacts (will be regenerated)
# ============================================================================
if [ -d "OrbbecSDK/build" ]; then
    echo "🔨 Removing build artifacts..."
    rm -rf OrbbecSDK/build/
fi

# ============================================================================
# 4. Remove large dataset (can re-download from Roboflow)
# ============================================================================
if [ -d "OrbbecSDK/Red_Ball_Detection.v1-v1.yolov8" ]; then
    echo "📊 Removing dataset (can re-download)..."
    rm -rf OrbbecSDK/Red_Ball_Detection.v1-v1.yolov8/
fi

# ============================================================================
# 5. Remove captured images (generated data)
# ============================================================================
if [ -d "OrbbecSDK/ball_captures" ]; then
    echo "📸 Removing ball captures..."
    rm -rf OrbbecSDK/ball_captures/
fi

# ============================================================================
# 6. Remove YOLO training outputs (large files)
# ============================================================================
if [ -d "runs" ]; then
    echo "🤖 Removing YOLO training runs..."
    rm -rf runs/
fi

# ============================================================================
# 7. Remove Python virtual environment (can recreate)
# ============================================================================
if [ -d "yolo_venv" ]; then
    echo "🐍 Removing Python virtual environment..."
    rm -rf yolo_venv/
fi

# ============================================================================
# 8. Remove YOLO pretrained model (can re-download)
# ============================================================================
if [ -f "yolov8n.pt" ]; then
    echo "🎯 Removing pretrained YOLOv8 model..."
    rm -f yolov8n.pt
fi

# ============================================================================
# 9. Remove macOS system files
# ============================================================================
echo "🍎 Removing macOS system files..."
find . -name ".DS_Store" -type f -delete
find . -name "._*" -type f -delete

# ============================================================================
# 10. Remove Python cache files
# ============================================================================
echo "🗑️  Removing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
find . -name "*.pyo" -delete 2>/dev/null

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "✅ Cleanup complete!"
echo ""
echo "📦 What was KEPT (essential files):"
echo "   ✓ OrbbecSDK/lib/          - SDK libraries"
echo "   ✓ OrbbecSDK/include/      - SDK headers"
echo "   ✓ OrbbecSDK/cmake/        - Build config"
echo "   ✓ OrbbecSDK/examples/cpp/Sample-ColorDepthFallback/"
echo "   ✓ OrbbecSDK/examples/cpp/Sample-TrainingDataCollector/"
echo "   ✓ Your Python scripts and documentation"
echo ""
echo "🗑️  What was REMOVED (can regenerate/re-download):"
echo "   ✗ OrbbecSDK/doc/          - Documentation"
echo "   ✗ OrbbecSDK/misc/         - Misc files"
echo "   ✗ OrbbecSDK/build/        - Build artifacts"
echo "   ✗ Dataset and captures    - Large data files"
echo "   ✗ YOLO runs/              - Training outputs"
echo "   ✗ yolo_venv/              - Virtual environment"
echo ""
echo "📊 Current project size:"
du -sh .
echo ""
echo "💡 Next steps:"
echo "   1. Rebuild: cd OrbbecSDK && cmake -B build && cmake --build build"
echo "   2. Recreate venv: python3 -m venv yolo_venv && source yolo_venv/bin/activate"
echo "   3. Install deps: pip install ultralytics opencv-python"
echo ""
