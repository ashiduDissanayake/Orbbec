#!/bin/bash
# Run HSV Color-Based Ball Detection System (Original)

cd "$(dirname "$0")/OrbbecSDK"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║          🎨 HSV Ball Detection System (Original)              ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check if executable exists
if [ ! -f "./build/bin/ColorDepthFallback" ]; then
    echo "❌ Executable not found. Building..."
    cmake --build build --target ColorDepthFallback
    
    if [ $? -ne 0 ]; then
        echo "❌ Build failed!"
        exit 1
    fi
fi

echo "✅ Starting HSV Ball Detection System..."
echo "   Press 'Q' or ESC to quit"
echo ""

# Set library path for macOS
if [ "$(uname)" == "Darwin" ]; then
    export DYLD_LIBRARY_PATH="$(pwd)/lib/macOS:$DYLD_LIBRARY_PATH"
    echo "📚 Library path: $(pwd)/lib/macOS"
else
    export LD_LIBRARY_PATH="$(pwd)/lib/linux_x64:$LD_LIBRARY_PATH"
fi

# Run the executable from build/bin directory
cd build/bin
./ColorDepthFallback

echo ""
echo "✅ Session ended"
