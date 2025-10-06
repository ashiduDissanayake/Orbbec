#!/usr/bin/env python3
"""
FAST Ball-Only Detection - Optimized for Speed
Only detects sports balls (class 32), ignores everything else
"""

from ultralytics import YOLO
import cv2

def main():
    print("\n╔══════════════════════════════════════════════╗")
    print("║   ⚡ FAST BALL DETECTION (Ball-Only)         ║")
    print("╚══════════════════════════════════════════════╝\n")
    
    # Load YOLO model
    print("📥 Loading YOLOv8 model...")
    model = YOLO('yolov8n.pt')
    print("✅ Model loaded!\n")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera!")
        return
    
    print("🎥 Camera opened. Detecting ONLY balls...")
    print("   Press 'Q' to quit\n")
    
    frame_count = 0
    import time
    fps_start = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # ⚡ OPTIMIZATION 1: Only detect class 32 (sports ball)
        # This makes it MUCH faster!
        results = model(frame, classes=[32], conf=0.15, verbose=False)
        
        # Get detections
        balls = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                balls.append({
                    'confidence': conf,
                    'bbox': (x1, y1, x2-x1, y2-y1),
                    'center': ((x1+x2)//2, (y1+y2)//2)
                })
        
        # Draw detections
        annotated = frame.copy()
        
        for ball in balls:
            x, y, w, h = ball['bbox']
            conf = ball['confidence']
            cx, cy = ball['center']
            
            # Green box for ball
            cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)
            
            # Label
            label = f"BALL {conf*100:.0f}%"
            cv2.putText(annotated, label, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Calculate FPS
        if frame_count % 30 == 0:
            fps = 30 / (time.time() - fps_start)
            fps_start = time.time()
            
            print(f"Frame {frame_count}: {len(balls)} ball(s) | FPS: {fps:.1f}")
            for i, ball in enumerate(balls):
                print(f"  Ball #{i+1}: {ball['confidence']*100:.1f}% conf")
        
        # Status text
        if balls:
            status = f"✅ {len(balls)} BALL(S) DETECTED"
            color = (0, 255, 0)
        else:
            status = "⚪ NO BALLS"
            color = (200, 200, 200)
        
        cv2.putText(annotated, status, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        cv2.imshow('FAST Ball Detection', annotated)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Test complete")

if __name__ == "__main__":
    main()