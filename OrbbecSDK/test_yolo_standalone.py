#!/usr/bin/env python3
"""
Test YOLO model directly on webcam to verify it works
"""

import cv2
from ultralytics import YOLO
import numpy as np

def test_yolo():
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║          YOLO MODEL VERIFICATION TEST                    ║")
    print("╚══════════════════════════════════════════════════════════╝\n")
    
    # Try both .pt and .onnx
    model_paths = [
        'runs/train/red_ball_v1/weights/best.pt',
        'best.onnx',
        '~/Dev/Orbbec/OrbbecSDK/build/bin/best.onnx'
    ]
    
    model = None
    for path in model_paths:
        try:
            import os
            expanded_path = os.path.expanduser(path)
            if os.path.exists(expanded_path):
                print(f"✅ Loading model: {expanded_path}")
                model = YOLO(expanded_path)
                break
        except:
            continue
    
    if not model:
        print("❌ No model found!")
        return
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open webcam!")
        return
    
    print("\n🎥 Webcam opened. Show the red ball to camera...")
    print("   Press 'Q' to quit\n")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run YOLO detection
        results = model(frame, conf=0.25, verbose=False)
        
        # Parse results
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                detections.append({
                    'conf': conf,
                    'bbox': (x1, y1, x2-x1, y2-y1),
                    'center': ((x1+x2)//2, (y1+y2)//2)
                })
        
        # Draw detections
        vis = frame.copy()
        
        if detections:
            for i, det in enumerate(detections):
                x, y, w, h = det['bbox']
                conf = det['conf']
                
                # Draw box
                cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 3)
                
                # Draw label
                label = f"Ball {conf*100:.0f}%"
                cv2.putText(vis, label, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            status = f"✅ DETECTING {len(detections)} ball(s)"
            color = (0, 255, 0)
        else:
            status = "❌ NO DETECTION"
            color = (0, 0, 255)
        
        # Show status
        cv2.putText(vis, status, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Print every 30 frames
        if frame_count % 30 == 0:
            if detections:
                print(f"Frame {frame_count}: ✅ {len(detections)} detection(s)")
                for i, det in enumerate(detections):
                    print(f"   [{i}] Confidence: {det['conf']*100:.1f}% at {det['center']}")
            else:
                print(f"Frame {frame_count}: ❌ No detections")
        
        cv2.imshow('YOLO Model Test', vis)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n✅ Test complete")

if __name__ == "__main__":
    test_yolo()