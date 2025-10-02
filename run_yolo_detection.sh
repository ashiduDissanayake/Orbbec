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
    
    # Check if build directory exists, if not configure CMake first
    if [ ! -d "./build" ]; then
        echo "🔧 Configuring CMake..."
        cmake -B build
        if [ $? -ne 0 ]; then
            echo "❌ CMake configuration failed!"
            exit 1
        fi
    fi
    
    # Build the target
    echo "🔨 Building ColorDepthFallback_YOLO..."
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

# Set library path for macOS
if [ "$(uname)" == "Darwin" ]; then
    export DYLD_LIBRARY_PATH="$(pwd)/lib/macOS:$DYLD_LIBRARY_PATH"
else
    export LD_LIBRARY_PATH="$(pwd)/lib/linux_x64:$LD_LIBRARY_PATH"
fi

# Run the executable from build/bin directory so relative paths work
cd build/bin
./ColorDepthFallback_YOLO

echo ""
echo "✅ Session ended"
