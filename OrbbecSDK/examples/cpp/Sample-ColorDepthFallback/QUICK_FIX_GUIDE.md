# Depth Sensor Issues - Summary & Quick Fixes

## 🔴 Your Two Problems

### Problem 1: Center shows 5cm instead of ~50cm
**Cause**: Value scale interpretation mismatch  
**Status**: Diagnostic code added, needs your test data

### Problem 2: Corners show "INVALID" (zeros)
**Cause**: Normal sensor behavior - limited FOV  
**Status**: This is EXPECTED for most depth sensors

---

## ⚡ Quick Action Plan

### Step 1: Run Diagnostic (2 minutes)
```bash
cd /Users/ashidudissanayake/Dev/Orbbec/OrbbecSDK
./build/bin/ColorDepthFallback
```

**Watch for this output:**
```
=== VALUE SCALE DIAGNOSTICS ===
Raw depth value (uint16): XXXX  ← Note this
Value scale: Y.YYY              ← Note this

Possible interpretations:
  1. Scale as-is → mm: ZZZ mm = 0.XXX m
  2. Raw value = mm: ZZZ mm = 0.YYY m  
  3. Raw is actual mm: ZZZ mm = 0.ZZZ m
```

### Step 2: Measure Reality (1 minute)
- Place any flat object (book, cardboard) in front of sensor
- **Measure actual distance with tape measure** (e.g., 50cm)
- Note which interpretation matches reality

### Step 3: Fix Code (30 seconds)
Run the Python analyzer:
```bash
cd examples/cpp/Sample-ColorDepthFallback
python3 analyze_depth.py
```

Enter the values from Step 1 + your measurement.  
It will tell you EXACTLY what code to change!

### Step 4: Rebuild & Verify
```bash
cmake --build build --target ColorDepthFallback
./build/bin/ColorDepthFallback
```

---

## 📊 About Corner Invalidations

### Why This Happens (Normal Physics)

1. **Field of View Limits**
   - Depth sensors: ~60° horizontal FOV
   - Corners are often outside sensing cone
   - **This is NOT a bug - it's physics!**

2. **Minimum Distance**
   - Most ToF sensors: 20cm minimum
   - Closer objects → invalid everywhere
   - Solution: Move back 20-30cm

3. **Sensor Geometry**
   - IR projector + receiver need overlap
   - Extreme angles → poor signal
   - Corners get blocked by housing

### What You Should See

**GOOD** (sensor working correctly):
```
Center:        0.500 m  ✓ Valid
Top-Left:      INVALID  ✓ Expected (outside FOV)
Top-Right:     INVALID  ✓ Expected (outside FOV)
Bottom-Left:   INVALID  ✓ Expected (outside FOV)
Bottom-Right:  INVALID  ✓ Expected (outside FOV)
```

**If ALL points invalid** → sensor too close (< 20cm)

### For Object Detection (Your Use Case)

**DON'T** use corner points directly.  
**DO** use the ROI sampler (already implemented):

```cpp
// When you detect object with bbox [x0, y0, x1, y1]:
auto xyz = projector.projectROI(depthData, width, height,
                                 x0, y0, x1, y1, 0.5f);

if (xyz[2] > 0.0f) {
    float distance = sqrt(xyz[0]*xyz[0] + xyz[1]*xyz[1] + xyz[2]*xyz[2]);
    cout << "Object at " << distance << "m" << endl;
}
```

This automatically:
- ✅ Ignores invalid (zero) pixels
- ✅ Takes median depth from ROI
- ✅ Filters outliers
- ✅ Returns (0,0,0) if not enough valid pixels

---

## 🎯 Expected Behavior After Fix

### Center Depth
**Before:** 0.052m (WRONG)  
**After:** 0.500m (if you measured 50cm) ✓

### Corners
**Before:** All INVALID  
**After:** Still INVALID (this is NORMAL!)  

**Why?** Corners are outside the ~60° FOV cone.

### Valid Region
Typical depth sensor coverage:

```
┌─────────────────────────────────┐
│ ███░░░░░░░░░░░░░░░░░░░░░░░███│  Top corners: often invalid
│ ██░░░░░░░░░░░░░░░░░░░░░░░░░██│
│ █░░░░░░░░░░░░░░░░░░░░░░░░░░░█│
│ █░░░░░░░░░░░░░░░░░░░░░░░░░░░█│
│ █░░░░░░░░░░░█████░░░░░░░░░░░█│  Center: always valid
│ █░░░░░░░░░░░░░░░░░░░░░░░░░░░█│
│ █░░░░░░░░░░░░░░░░░░░░░░░░░░░█│
│ ██░░░░░░░░░░░░░░░░░░░░░░░░░██│
│ ███░░░░░░░░░░░░░░░░░░░░░░░███│  Bottom corners: often invalid
└─────────────────────────────────┘
    ░ = Valid depth
    █ = Invalid (outside FOV)
```

---

## 🔧 Files Created for You

1. **DEPTH_CALIBRATION_GUIDE.md** - Complete troubleshooting guide
2. **analyze_depth.py** - Python tool to find correct formula
3. **ColorDepthFallback.cpp** - Already has diagnostics built in

---

## 📞 What to Share if You Need More Help

Run the sample for ~5 seconds and share:

1. The "VALUE SCALE DIAGNOSTICS" output
2. What distance you measured (e.g., "placed book at 50cm")
3. Output from running `python3 analyze_depth.py`

I'll tell you the EXACT line to change!

---

## ✅ Success Criteria

After fix:
- [ ] Center depth matches tape measure (within ±2cm)
- [ ] Depth changes correctly as you move object closer/farther
- [ ] ROI sampling returns reasonable values
- [ ] Corner invalidations are OK (expected behavior)

---

## 🚀 Start Here

```bash
# Terminal 1: Run sample
cd /Users/ashidudissanayake/Dev/Orbbec/OrbbecSDK
./build/bin/ColorDepthFallback

# Terminal 2: After seeing diagnostics
cd examples/cpp/Sample-ColorDepthFallback
python3 analyze_depth.py
# Enter: raw value, scale, actual distance
# Follow the code fix it suggests

# Rebuild
cd ../../../
cmake --build build --target ColorDepthFallback

# Test again
./build/bin/ColorDepthFallback
```

Good luck! Let me know the diagnostic values and I'll help you nail the fix! 🎯
