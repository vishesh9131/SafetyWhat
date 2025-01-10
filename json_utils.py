import cv2
import os
import sys
import argparse
import time
import torch 
from image_retriever import ImageRetriever
import warnings
import logging
import json
from datetime import datetime
import glob

def format_detection_json(main_obj, sub_objs):
    return {
        "object": main_obj["name"],
        "id": main_obj["id"],
        "bbox": main_obj["bbox"],
        "confidence": main_obj["confidence"],
        "subobjects": [
            {
                "object": sub["name"],
                "id": sub["id"],
                "bbox": sub["bbox"],
                "confidence": sub["confidence"]
            } for sub in sub_objs
        ]
    }

def print_detection_info(detections):
    """Updated print function to show subobjects"""
    print("\033[94m╭── Detections ────────────────────────────╮")
    for det in detections:
        print(f"│ {det['object']:<15} ID: {det['id']:<3}          │")
        for subobj in det.get('subobjects', []):
            print(f"│   └─ {subobj['object']:<13} ID: {subobj['id']:<3}           │")
    print("╰─────────────────────────────────────────╯\033[0m")

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

def is_duplicate(det, unique_objects, iou_threshold=0.5):
    """Check if a detection is a duplicate based on IoU."""
    for obj in unique_objects:
        if det['object'] == obj['object']:
            iou = calculate_iou(det['bbox'], obj['bbox'])
            if iou > iou_threshold:
                return True
    return False

def merge_and_cleanup_json_files(output_dir='.', merged_filename='final_output.json'):
    """Merge all JSON files in the output directory and delete them after merging."""
    json_files = glob.glob(os.path.join(output_dir, 'detections_*.json'))
    all_detections = []
    
    # First, collect all detections
    for json_file in json_files:
        with open(json_file, 'r') as f:
            detections = json.load(f)
            all_detections.extend(detections)
    
    # Sort by number of subobjects (non-empty first) and confidence
    all_detections.sort(key=lambda x: (-len(x.get('subobjects', [])), -x.get('confidence', 0)))
    
    # Remove duplicates while keeping ones with subobjects
    unique_detections = []
    seen_objects = {}  # Dictionary to track objects by class
    
    for det in all_detections:
        obj_class = det['object']
        has_subobjects = len(det.get('subobjects', [])) > 0
        
        # If we haven't seen this class or this one has subobjects and the previous didn't
        if obj_class not in seen_objects:
            seen_objects[obj_class] = det
            unique_detections.append(det)
        elif has_subobjects and not len(seen_objects[obj_class].get('subobjects', [])) > 0:
            # Replace the existing detection with this one that has subobjects
            unique_detections.remove(seen_objects[obj_class])
            seen_objects[obj_class] = det
            unique_detections.append(det)
        elif has_subobjects and calculate_iou(det['bbox'], seen_objects[obj_class]['bbox']) < 0.5:
            # If it has subobjects and is in a different location, keep it
            unique_detections.append(det)
            
    # Write merged detections to a single JSON file
    with open(os.path.join(output_dir, merged_filename), 'w') as f:
        json.dump(unique_detections, f, indent=4)

    # Delete individual JSON files
    for json_file in json_files:
        os.remove(json_file)


