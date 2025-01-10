from config import get_args
from helper import setup_environment, print_banner
from detection import initialize_model, process_video,process_detections_fast
from json_utils import merge_and_cleanup_json_files, print_detection_info
from mapper import SUBOBJECTS_MAP

import os
import json
from datetime import datetime
import cv2
import torch
import warnings
import sys
import time

def save_detections_json(detections, frame_number, output_dir='.'):
    """Save detections in JSON format"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"detections_{timestamp}_{frame_number}.json")
    
    json_data = []
    for det in detections:
        detection = {
            "object": det['object'],
            "id": det['id'],
            "bbox": det['bbox'],
            "confidence": float(det['confidence']),
            "subobjects": [
                {
                    "object": sub['object'],
                    "id": sub['id'],
                    "bbox": sub['bbox'],
                    "confidence": float(sub['confidence'])
                } for sub in det.get('subobjects', [])
            ]
        }
        json_data.append(detection)
    
    with open(output_file, 'w') as f:
        json.dump(json_data, f, indent=4)
    
    print(f"Saved detections to {output_file}")

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0

def save_subobject_images(frame, detections, output_dir='subobject_images'):
    """Save cropped images of subobjects"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for det in detections:
        for sub in det.get('subobjects', []):
            x1, y1, x2, y2 = map(int, sub['bbox'])
            sub_img = frame[y1:y2, x1:x2]
            sub_img_path = os.path.join(output_dir, f"{sub['object']}_{sub['id']}.png")
            cv2.imwrite(sub_img_path, sub_img)
            print(f"Saved subobject image to {sub_img_path}")

def main(use_webcam=False, conf_threshold=0.3, iou_threshold=0.3, max_det=100, img_size=640, frame_skip=1):
    args = get_args()
    setup_environment()
    print_banner()
    
    model = initialize_model(args.conf_threshold, args.iou_threshold, args.max_det, args.img_size, args.model)
    
    # Initialize variables before the video loop
    frame_count = 0
    prev_time = time.time()
    fps = 0
    
    cap = cv2.VideoCapture(args.video if not args.webcam else 1)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    def save_detections_json(detections, frame_number):
        """Save detections in JSON format"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"detections_{timestamp}_{frame_number}.json"
        
        json_data = []
        for det in detections:
            detection = {
                "object": det['object'],
                "id": det['id'],
                "bbox": det['bbox'],
                "confidence": float(det['confidence']),
                "subobjects": [
                    {
                        "object": sub['object'],
                        "id": sub['id'],
                        "bbox": sub['bbox'],
                        "confidence": float(sub['confidence'])
                    } for sub in det.get('subobjects', [])
                ]
            }
            json_data.append(detection)
        
        with open(output_file, 'w') as f:
            json.dump(json_data, f, indent=4)

    def draw_label_with_background(frame, text, position, font_scale=0.5, thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX):
        """Draw text with filled background"""
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Calculate background rectangle coordinates
        x, y = position
        background_coords = (
            (x, y - text_height - baseline),
            (x + text_width, y + baseline)
        )
        
        # Draw background rectangle
        cv2.rectangle(frame, background_coords[0], background_coords[1], (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), thickness)

    while cap.isOpened():
        try:
            ret, frame = cap.read()
            if not ret:
                print("\n\033[93m[System]\033[0m End of video or frame read error")
                break

            # Skip frames to increase FPS
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue

            # Calculate FPS
            current_time = time.time()
            if current_time - prev_time >= 1.0:
                fps = frame_count
                frame_count = 0
                prev_time = current_time

            # Process frame
            results = model(frame)
            
            # Convert results to proper format
            detections = []
            for *xyxy, conf, cls in results.xyxy[0]:
                detection = {
                    'xmin': float(xyxy[0]),
                    'ymin': float(xyxy[1]),
                    'xmax': float(xyxy[2]),
                    'ymax': float(xyxy[3]),
                    'confidence': float(conf),
                    'name': model.names[int(cls)]
                }
                detections.append(detection)
            
            # Process detections to find subobjects
            processed_detections = process_detections_fast(detections, SUBOBJECTS_MAP)
            
            # Save detections to JSON every 30 frames
            if frame_count % 30 == 0:
                save_detections_json(processed_detections, frame_count)
            
            # Save subobject images
            save_subobject_images(frame, processed_detections)
            
            # Display results
            print_detection_info(processed_detections)
            
            # Draw FPS on frame
            draw_label_with_background(frame, f"FPS: {fps}", (10, 30))
            
            # Draw detections on frame
            for det in processed_detections:
                # Draw main object
                x1, y1, x2, y2 = map(int, det['bbox'])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw main object label with background
                label = f"{det['object']} {det['confidence']:.2f}"
                draw_label_with_background(frame, label, (x1, y1-10))
                
                # Draw subobjects
                for sub in det.get('subobjects', []):
                    x1, y1, x2, y2 = map(int, sub['bbox'])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    
                    # Draw subobject label with background
                    sub_label = f"{sub['object']} {sub['confidence']:.2f}"
                    draw_label_with_background(frame, sub_label, (x1, y1-10))
            
            # Show frame
            cv2.imshow('SAFETYWHAT ASSESSMENT', frame)
            
            frame_count += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        except Exception as e:
            print(f"\n\033[91m[System]\033[0m Error: {str(e)}")
            continue

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    merge_and_cleanup_json_files()

if __name__ == "__main__":
    main()