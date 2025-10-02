#!/usr/bin/env python3
"""
Quick Depth Value Analyzer
Run this after collecting some depth readings from the sensor
"""

def analyze_depth_reading(raw_value, value_scale, actual_distance_cm):
    """
    Analyze which interpretation of depth matches reality
    
    Args:
        raw_value: uint16 raw depth value from sensor
        value_scale: float from getValueScale()
        actual_distance_cm: measured distance in centimeters
    """
    actual_m = actual_distance_cm / 100.0
    
    print(f"\n{'='*60}")
    print(f"DEPTH VALUE ANALYSIS")
    print(f"{'='*60}")
    print(f"Measured distance: {actual_distance_cm} cm ({actual_m:.3f} m)")
    print(f"Raw sensor value:  {raw_value}")
    print(f"Value scale:       {value_scale}")
    print(f"\n{'Interpretation':<30} {'Result':<15} {'Error'}")
    print(f"{'-'*60}")
    
    interpretations = [
        ("Raw * scale * 0.001", raw_value * value_scale * 0.001),
        ("Raw * 0.001 (ignore scale)", raw_value * 0.001),
        ("Raw / scale * 0.001", raw_value / value_scale * 0.001 if value_scale != 0 else 0),
        ("Raw / 1000 / scale", raw_value / 1000.0 / value_scale if value_scale != 0 else 0),
        ("Raw = centimeters", raw_value / 100.0),
        ("Scale is mm→m, Raw as-is", raw_value * value_scale),
    ]
    
    best_match = None
    best_error = float('inf')
    
    for name, result_m in interpretations:
        error = abs(result_m - actual_m)
        error_pct = (error / actual_m * 100) if actual_m > 0 else 0
        
        marker = ""
        if error < best_error:
            best_error = error
            best_match = name
            marker = " ← BEST"
        
        print(f"{name:<30} {result_m:>6.3f} m     {error:>5.3f} m ({error_pct:>5.1f}%){marker}")
    
    print(f"\n{'='*60}")
    print(f"RECOMMENDATION: Use '{best_match}'")
    print(f"{'='*60}\n")
    
    return best_match


def suggest_code_fix(interpretation):
    """Print the C++ code fix based on interpretation"""
    print("CODE FIX:")
    print("=" * 60)
    print("In DepthProjector::project(), change line:")
    print("    const float z = depthRaw * valueScale_ * 0.001f;")
    print("\nTo:\n")
    
    if "ignore scale" in interpretation.lower():
        print("    const float z = depthRaw * 0.001f;  // Raw is millimeters")
    elif "raw / scale" in interpretation.lower():
        print("    const float z = (depthRaw / valueScale_) * 0.001f;  // Inverted scale")
    elif "centimeters" in interpretation.lower():
        print("    const float z = depthRaw * 0.01f;  // Raw is centimeters")
    elif "scale is mm" in interpretation.lower():
        print("    const float z = depthRaw * valueScale_;  // Scale converts to meters directly")
    else:
        print("    // Keep current formula")
        print("    const float z = depthRaw * valueScale_ * 0.001f;")
    
    print("=" * 60)


# Example usage
if __name__ == "__main__":
    print("\nDEPTH SENSOR CALIBRATION TOOL")
    print("=" * 60)
    print("Instructions:")
    print("1. Place object at KNOWN distance (measure with tape)")
    print("2. Run ColorDepthFallback and note the diagnostic output")
    print("3. Enter values below")
    print("=" * 60)
    
    try:
        raw_value = int(input("\nEnter raw depth value (uint16): "))
        value_scale = float(input("Enter value scale (float): "))
        actual_cm = float(input("Enter actual measured distance (cm): "))
        
        best = analyze_depth_reading(raw_value, value_scale, actual_cm)
        suggest_code_fix(best)
        
        print("\n✅ NEXT STEPS:")
        print("1. Apply the code fix above")
        print("2. Rebuild: cmake --build build --target ColorDepthFallback")
        print("3. Test again and verify distance is now correct")
        print()
        
    except KeyboardInterrupt:
        print("\n\nCancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nExample values:")
        print("  Raw value: 520  (for ~50cm)")
        print("  Value scale: 1.0")
        print("  Actual distance: 50")
