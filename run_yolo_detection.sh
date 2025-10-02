#!/bin/bash
# Run YOLO Ball Detection System

cd "$(dirname "$0")/OrbbecSDK"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║          🎯 YOLO Ball Detection & Tracking System              ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check if executable exists
if [ ! -f "./build/bin/ColorDepthFallback_YOLO" ]; then
    echo "❌ Executable not found. Building..."
    cmake --build build --target ColorDepthFallback_YOLO
    
    if [ $? -ne 0 ]; then
        echo "❌ Build failed!"
        exit 1
    fi
fi

# Check if model exists
if [ ! -f "./examples/cpp/Sample-ColorDepthFallback/best.onnx" ] && [ ! -f "./build/bin/best.onnx" ]; then
    echo "❌ ONNX model not found!"
    echo "Please export your model first:"
    echo "  cd .."
    echo "  python3 export_yolo_to_onnx.py"
    exit 1
fi

echo "✅ Starting YOLO Ball Detection System..."
echo "   Press 'Q' or ESC to quit"
echo ""

# Run the executable
./build/bin/ColorDepthFallback_YOLO

echo ""
echo "✅ Session ended"
