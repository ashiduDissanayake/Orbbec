# Depth Sensor Calibration & Troubleshooting Guide

## 🔴 Problem 1: Incorrect Depth Readings (5cm instead of 50cm)

### Root Causes

#### A. Value Scale Misinterpretation
The Orbbec SDK's `getValueScale()` returns a multiplier that can be interpreted differently:

1. **Scale = 1.0**: Raw values are already in millimeters
   - `depthMeters = rawValue * 0.001`
2. **Scale = 0.1 or 0.01**: Raw values need scaling
   - `depthMeters = rawValue * scale * 0.001`
3. **Scale represents unit size**: Less common
   - `depthMeters = rawValue / (scale * 1000)`

**Our current formula:**
```cpp
float depthMeters = rawDepth * valueScale * 0.001f;
```

#### B. Incorrect Firmware/Driver Configuration
- Depth mode set incorrectly (e.g., "close range" vs "far range")
- Wrong depth unit in device settings

### Diagnostic Steps

1. **Run the diagnostic output** (already added to code):
   ```
   === VALUE SCALE DIAGNOSTICS ===
   Raw depth value (uint16): XXXX
   Value scale: Y.YYY
   
   Possible interpretations:
     1. Scale as-is → mm: ... 
     2. Raw value = mm: ...
     3. Raw is actual mm: ...
   ```

2. **Physical Measurement Test**:
   - Place an object at a KNOWN distance (use tape measure)
   - Example: 50cm from sensor
   - Compare all three interpretations
   - Choose the one matching reality

3. **Check typical values**:
   - If `rawDepth ≈ 500` for 50cm → raw is in millimeters (scale ignored)
   - If `rawDepth ≈ 5000` for 50cm AND `scale = 0.1` → use formula as-is
   - If `rawDepth ≈ 50` for 50cm → raw might be in centimeters

### Solutions

#### Solution 1: Ignore Value Scale (Most Common)
If raw values are already in millimeters:

```cpp
// In DepthProjector::project()
const float z = depthRaw * 0.001f;  // Remove valueScale multiplication
```

#### Solution 2: Adjust Scale Interpretation
If scale needs division instead of multiplication:

```cpp
const float z = (depthRaw / valueScale_) * 0.001f;
```

#### Solution 3: Add Manual Calibration Factor
After testing with known distances:

```cpp
const float CALIBRATION_FACTOR = 10.0f;  // Adjust based on tests
const float z = (depthRaw * valueScale_ * CALIBRATION_FACTOR) * 0.001f;
```

---

## 🔴 Problem 2: Corner Depth Invalidations (All Zeros)

### Root Causes

#### A. Limited Field of View (FOV)
Depth sensors have:
- **Horizontal FOV**: Typically 50-75°
- **Vertical FOV**: Typically 40-60°
- **Corners are often outside valid sensing area**

#### B. Minimum Distance Constraint
- Most ToF/Structured Light sensors: **min distance ~20cm**
- If object too close → invalid depth

#### C. Sensor Physics
- **IR projector/receiver geometry**: Corners may not receive enough pattern
- **Low reflectivity**: Dark surfaces at steep angles
- **Occlusions**: Sensor housing blocks extreme corners

### Validation Steps

1. **Check Valid Depth Percentage**:
```cpp
// Count valid pixels
int validCount = 0;
for (uint32_t i = 0; i < width * height; ++i) {
    if (data[i] > 0) validCount++;
}
float validPercent = 100.0f * validCount / (width * height);
std::cout << "Valid depth pixels: " << validPercent << "%" << std::endl;
```

2. **Visualize Valid Region**:
   - The `visualizeDepthWithOverlay()` function already does this
   - Black pixels = invalid
   - Colored pixels = valid depth

3. **Test at Different Distances**:
   - **< 20cm**: Most sensors fail → all invalid
   - **20-50cm**: Partial coverage, corners may fail
   - **50cm-3m**: Optimal range, most pixels valid
   - **> 3m**: Signal weakens, corners fail first

### Solutions

#### Solution 1: Use Center Region Only
For object detection, avoid corners:

```cpp
// Sample from center 50% of image
int margin = width / 4;  // 25% margin on each side
for (int y = margin; y < height - margin; ++y) {
    for (int x = margin; x < width - margin; ++x) {
        uint16_t depth = data[y * width + x];
        if (depth > 0) {
            // Process valid depth
        }
    }
}
```

#### Solution 2: Use ROI Sampling (Already Implemented)
```cpp
// Sample from bounding box with percentile filtering
auto xyz = projector.projectROI(data, width, height, 
                                 x0, y0, x1, y1, 
                                 0.5f);  // Median depth
```
This ignores invalid pixels automatically!

#### Solution 3: Depth Inpainting
Fill invalid regions using interpolation:

```cpp
void inpaintDepth(uint16_t* data, int width, int height) {
    for (int y = 1; y < height-1; ++y) {
        for (int x = 1; x < width-1; ++x) {
            if (data[y*width + x] == 0) {
                // Average valid neighbors
                int sum = 0, count = 0;
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        uint16_t d = data[(y+dy)*width + (x+dx)];
                        if (d > 0) { sum += d; count++; }
                    }
                }
                if (count > 0) data[y*width + x] = sum / count;
            }
        }
    }
}
```

---

## 📊 Calibration Procedure

### Step 1: Value Scale Calibration

1. **Setup**:
   - Place a flat surface (cardboard, wall) at **exactly 50cm**
   - Point sensor perpendicular to surface
   - Ensure good lighting

2. **Run diagnostic**:
   ```bash
   ./build/bin/ColorDepthFallback
   ```
   
3. **Record output**:
   ```
   Raw depth value (uint16): ____
   Value scale: ____
   
   Interpretation 1: ____ m
   Interpretation 2: ____ m
   Interpretation 3: ____ m
   ```

4. **Compare to reality** (50cm = 0.5m):
   - Whichever interpretation is closest → use that formula

5. **Update code** in `DepthProjector::project()`:
   ```cpp
   // Option A: Raw is millimeters
   const float z = depthRaw * 0.001f;
   
   // Option B: Use scale as-is (current)
   const float z = depthRaw * valueScale_ * 0.001f;
   
   // Option C: Scale is inverted
   const float z = (depthRaw / valueScale_) * 0.001f;
   ```

### Step 2: Intrinsics Validation

1. **Check intrinsics printout**:
   ```
   === Depth Intrinsics ===
     fx: 554.256
     fy: 554.256
     cx: 320
     cy: 240
   ```

2. **Validate FOV**:
   ```cpp
   float fovH = 2.0f * atan(intrinsic.width / (2.0f * intrinsic.fx)) * 180.0f / M_PI;
   float fovV = 2.0f * atan(intrinsic.height / (2.0f * intrinsic.fy)) * 180.0f / M_PI;
   std::cout << "FOV Horizontal: " << fovH << "°" << std::endl;
   std::cout << "FOV Vertical: " << fovV << "°" << std::endl;
   ```
   
   **Expected**: 50-75° horizontal, 40-60° vertical

3. **If FOV seems wrong** → intrinsics are incorrect:
   - Try acquiring from device calibration
   - Use default approximation (current fallback)
   - Manually measure and set

### Step 3: Corner Coverage Test

1. **Place objects at known distances**:
   - Center: 50cm
   - Top-left quadrant: 60cm
   - Top-right quadrant: 60cm
   - etc.

2. **Run test and observe**:
   ```
   ╔════════════════════════════════════════════════════════╗
   ║        DEPTH PROJECTION TEST RESULTS                   ║
   ╠════════════════════════════════════════════════════════╣
   ║ Point       │ Pixel   │ Depth (m) │ XYZ (meters)      ║
   ╠════════════════════════════════════════════════════════╣
   ║ Center      │ (320,240) │ 0.500   │ (0, 0, 0.5)      ║
   ║ Top-Left    │ (160,120) │ INVALID │                   ║  ← Common
   ```

3. **Expected behavior**:
   - Center always valid (unless too close/far)
   - Corners may be invalid depending on FOV
   - **This is NORMAL** for many sensors!

---

## 🎯 For Your Object Detection Use Case

### Recommended Approach

1. **Use ROI-based depth sampling** (already implemented):
   ```cpp
   // When you detect an object with bounding box [x0, y0, x1, y1]:
   auto xyz = projector.projectROI(depthData, width, height,
                                     x0, y0, x1, y1,
                                     0.5f);  // Median depth
   
   if (xyz[2] > 0.0f) {
       float distance = sqrt(xyz[0]*xyz[0] + xyz[1]*xyz[1] + xyz[2]*xyz[2]);
       std::cout << "Object at " << distance << " meters" << std::endl;
   }
   ```

2. **Filter by valid pixel percentage**:
   ```cpp
   // In projectROI, add:
   float validRatio = validDepths.size() / (float)((x1-x0) * (y1-y0));
   if (validRatio < 0.3f) {
       // Less than 30% valid → unreliable
       return {0.f, 0.f, 0.f};
   }
   ```

3. **Ignore detections near image borders**:
   ```cpp
   int margin = 50;  // pixels
   if (x0 < margin || y0 < margin || 
       x1 > width-margin || y1 > height-margin) {
       // Too close to edge → skip or flag as unreliable
   }
   ```

---

## 🔧 Quick Fixes to Try Now

### Fix 1: Test Raw Values Only
```cpp
// Temporarily modify DepthProjector::project()
const float z = depthRaw * 0.001f;  // Ignore valueScale
```
Rebuild and test. If distances now correct → **value scale was the issue**

### Fix 2: Reduce Test Points
```cpp
std::vector<DepthTestPoint> testPoints = {
    {"Center", static_cast<int>(width / 2), static_cast<int>(height / 2)},
    {"Near-Center-1", static_cast<int>(width / 2) - 50, static_cast<int>(height / 2)},
    {"Near-Center-2", static_cast<int>(width / 2) + 50, static_cast<int>(height / 2)},
    {"Near-Center-3", static_cast<int>(width / 2), static_cast<int>(height / 2) - 50},
    {"Near-Center-4", static_cast<int>(width / 2), static_cast<int>(height / 2) + 50}
};
```
Test points closer to center are more likely to have valid depth

### Fix 3: Add Depth Range Logging
```cpp
// After acquiring frame
uint16_t minD = 65535, maxD = 0;
int validCnt = 0;
for (uint32_t i = 0; i < width * height; ++i) {
    if (data[i] > 0) {
        minD = std::min(minD, data[i]);
        maxD = std::max(maxD, data[i]);
        validCnt++;
    }
}
std::cout << "Depth range: " << minD << " - " << maxD << " (raw)" << std::endl;
std::cout << "Valid pixels: " << (100.0f * validCnt / (width * height)) << "%" << std::endl;
```

---

## 📝 Report Template

After running tests, fill this in:

```
DEPTH CALIBRATION REPORT
========================

Sensor: Orbbec Astra _______
Distance tested: 50cm (0.5m)

Value Scale Diagnostics:
- Raw depth value: _______
- Value scale: _______
- Interpretation 1: _______ m
- Interpretation 2: _______ m  
- Interpretation 3: _______ m
- ACTUAL distance: 0.5m
- **Best match**: Interpretation ___ 

Intrinsics:
- fx: _______  fy: _______
- cx: _______  cy: _______
- Calculated FOV H: _______°  V: _______°
- Source: [ ] Device [ ] Default fallback

Corner Coverage:
- Center valid: [ ] Yes [ ] No
- Top-Left valid: [ ] Yes [ ] No
- Top-Right valid: [ ] Yes [ ] No
- Bottom-Left valid: [ ] Yes [ ] No
- Bottom-Right valid: [ ] Yes [ ] No
- Overall valid pixel %: _______%

CONCLUSION:
- Value scale formula: _______________________
- Usable depth range: _______ cm to _______ cm
- Recommended ROI margin: _______ pixels from edge
```

---

## 🚀 Next Steps

1. **Run the diagnostic** and note the value scale output
2. **Test with physical measurement** at 50cm
3. **Adjust the formula** in `DepthProjector::project()`
4. **Re-test and iterate**
5. **Document your findings** in the report template above

Let me know the diagnostic output and I'll help you pinpoint the exact fix!
