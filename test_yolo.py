
from ultralytics import YOLO
import cv2
import numpy as np
import time
from collections import deque

print("\n╔════════════════════════════════════════════════════════╗")
print("║   🎾 RED BALL DETECTOR - WEBCAM TEST                   ║")
print("╚════════════════════════════════════════════════════════╝\n")

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_PATH = 'best.pt'  # Your downloaded model
CONFIDENCE_THRESHOLD = 0.9  # Lower = more sensitive
WEBCAM_ID = 0  # 0 = default webcam, 1 = external
WINDOW_NAME = 'Red Ball Detection Test'

# Colors (BGR format)
COLOR_BOX = (0, 255, 0)      # Green box
COLOR_TEXT = (255, 255, 255)  # White text
COLOR_CONF = (0, 255, 255)    # Yellow confidence
COLOR_FPS = (255, 0, 255)     # Magenta FPS

# ============================================================
# LOAD MODEL
# ============================================================
print(f"📥 Loading model: {MODEL_PATH}")

try:
    model = YOLO(MODEL_PATH)
    print("✅ Model loaded successfully!\n")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    print("\n💡 Make sure 'red_ball_model.pt' is in the same folder")
    exit(1)

# ============================================================
# OPEN WEBCAM
# ============================================================
print(f"🎥 Opening webcam (ID: {WEBCAM_ID})...")

cap = cv2.VideoCapture(WEBCAM_ID)

if not cap.isOpened():
    print("❌ Failed to open webcam!")
    print("\n💡 Try changing WEBCAM_ID to 1 or 2")
    exit(1)

# Set resolution (optional - adjust for your camera)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"✅ Webcam opened: {width}×{height}\n")

# ============================================================
# FPS CALCULATION
# ============================================================
fps_queue = deque(maxlen=30)  # Average over 30 frames
prev_time = time.time()

# ============================================================
# STATISTICS
# ============================================================
frame_count = 0
total_detections = 0
detection_history = deque(maxlen=100)  # Last 100 frames

# ============================================================
# INSTRUCTIONS
# ============================================================
print("╔════════════════════════════════════════════════════════╗")
print("║   CONTROLS:                                            ║")
print("║   'Q' or 'ESC'  - Quit                                 ║")
print("║   'S'           - Save current frame                   ║")
print("║   'R'           - Reset statistics                     ║")
print("║   '+'           - Increase confidence threshold        ║")
print("║   '-'           - Decrease confidence threshold        ║")
print("╚════════════════════════════════════════════════════════╝\n")

print("🎯 Starting detection...\n")
print("Point your webcam at a ball (soccer/basketball/any round ball)")
print("Press 'Q' to quit\n")

# ============================================================
# MAIN DETECTION LOOP
# ============================================================
try:
    while True:
        # Read frame
        ret, frame = cap.read()
        
        if not ret:
            print("❌ Failed to read frame")
            break
        
        frame_count += 1
        
        # Run YOLO detection
        results = model.predict(
            frame,
            conf=CONFIDENCE_THRESHOLD,
            iou=0.45,
            verbose=False
        )
        
        # Extract detections
        boxes = results[0].boxes
        num_balls = len(boxes)
        total_detections += num_balls
        detection_history.append(num_balls)
        
        # Draw detections
        for idx, box in enumerate(boxes):
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            
            # Calculate center
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            
            # Calculate size
            w = x2 - x1
            h = y2 - y1
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BOX, 3)
            
            # Draw center point
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            
            # Draw crosshair
            cv2.line(frame, (cx - 10, cy), (cx + 10, cy), (0, 0, 255), 2)
            cv2.line(frame, (cx, cy - 10), (cx, cy + 10), (0, 0, 255), 2)
            
            # Label text
            label = f"Ball #{idx+1}"
            conf_text = f"{confidence*100:.1f}%"
            pos_text = f"({cx}, {cy})"
            size_text = f"{w}×{h}px"
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, 
                         (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0] + 10, y1),
                         COLOR_BOX, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Draw info
            info_y = y2 + 20
            cv2.putText(frame, conf_text, (x1, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_CONF, 2)
            cv2.putText(frame, pos_text, (x1, info_y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_TEXT, 1)
            cv2.putText(frame, size_text, (x1, info_y + 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_TEXT, 1)
        
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        fps_queue.append(fps)
        avg_fps = np.mean(fps_queue)
        
        # Draw overlay info panel
        overlay_height = 150
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, overlay_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # Status text
        y_offset = 25
        line_height = 25
        
        # Title
        cv2.putText(frame, "RED BALL DETECTOR", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += line_height
        
        # FPS
        fps_text = f"FPS: {avg_fps:.1f}"
        fps_color = (0, 255, 0) if avg_fps > 20 else (0, 165, 255) if avg_fps > 10 else (0, 0, 255)
        cv2.putText(frame, fps_text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)
        y_offset += line_height
        
        # Detections
        det_text = f"Balls: {num_balls}"
        det_color = (0, 255, 0) if num_balls > 0 else (128, 128, 128)
        cv2.putText(frame, det_text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, det_color, 2)
        y_offset += line_height
        
        # Frame count
        cv2.putText(frame, f"Frame: {frame_count}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += line_height
        
        # Confidence threshold
        cv2.putText(frame, f"Threshold: {CONFIDENCE_THRESHOLD:.2f}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Right side stats
        detection_rate = (sum(detection_history) / len(detection_history) * 100) if detection_history else 0
        stats_x = width - 180
        y_offset = 25
        
        cv2.putText(frame, f"Avg: {detection_rate:.1f}% frames", (stats_x, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += line_height
        cv2.putText(frame, f"Total: {total_detections} balls", (stats_x, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Status indicator
        if num_balls > 0:
            status_text = "BALL DETECTED"
            status_color = (0, 255, 0)
            cv2.circle(frame, (width - 20, 20), 10, status_color, -1)
        else:
            status_text = "NO DETECTION"
            status_color = (128, 128, 128)
            cv2.circle(frame, (width - 20, 20), 10, status_color, -1)
        
        # Show frame
        cv2.imshow(WINDOW_NAME, frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # Q or ESC
            print("\n⏹️  Stopping detection...")
            break
        
        elif key == ord('s'):  # Save frame
            filename = f"capture_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"📸 Saved: {filename}")
        
        elif key == ord('r'):  # Reset stats
            frame_count = 0
            total_detections = 0
            detection_history.clear()
            print("🔄 Statistics reset")
        
        elif key == ord('+') or key == ord('='):  # Increase threshold
            CONFIDENCE_THRESHOLD = min(0.95, CONFIDENCE_THRESHOLD + 0.05)
            print(f"📈 Threshold: {CONFIDENCE_THRESHOLD:.2f}")
        
        elif key == ord('-') or key == ord('_'):  # Decrease threshold
            CONFIDENCE_THRESHOLD = max(0.05, CONFIDENCE_THRESHOLD - 0.05)
            print(f"📉 Threshold: {CONFIDENCE_THRESHOLD:.2f}")

except KeyboardInterrupt:
    print("\n⚠️  Interrupted by user")

# ============================================================
# CLEANUP
# ============================================================
cap.release()
cv2.destroyAllWindows()

# ============================================================
# FINAL STATISTICS
# ============================================================
print("\n" + "="*60)
print("📊 SESSION SUMMARY")
print("="*60)
print(f"   Total frames: {frame_count}")
print(f"   Total detections: {total_detections}")
print(f"   Average FPS: {np.mean(fps_queue):.1f}")
if detection_history:
    detection_rate = (sum(detection_history) / len(detection_history) * 100)
    print(f"   Detection rate: {detection_rate:.1f}% of frames")
print("="*60)

print("\n✅ Test complete!")
print("\n💡 Next steps:")
print("   • If detection works well → Integrate with Orbbec depth camera")
print("   • If too sensitive → Increase confidence threshold")
print("   • If missing balls → Decrease confidence threshold")