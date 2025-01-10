import torch
import cv2
from video import open_video_capture, calculate_fps, draw_detections
import time

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

def initialize_model(conf_threshold, iou_threshold, max_det, img_size, model_type='yolov5s'):
    print("\033[94m[System]\033[0m Initializing model...")
    model = torch.hub.load('ultralytics/yolov5', model_type, pretrained=True, verbose=False)
    model.conf = conf_threshold
    model.iou = iou_threshold
    model.max_det = max_det
    model.imgsz = img_size

    if torch.cuda.is_available():
        model.cuda()
        print("\033[92m[SUCCESS]\033[0m CUDA acceleration enabled")
    
    return model

def process_detections_fast(detections, subobjects_map):
    """Process detections to find subobjects and assign unique IDs"""
    processed = []
    used_ids = set()
    
    # Convert detections to proper format with bbox
    formatted_detections = []
    for idx, det in enumerate(detections):
        formatted_det = {
            'id': idx + 1,
            'object': det['name'],
            'bbox': [
                float(det['xmin']),
                float(det['ymin']),
                float(det['xmax']),
                float(det['ymax'])
            ],
            'confidence': float(det['confidence']),
            'subobjects': det.get('subobjects', [])
        }
        formatted_detections.append(formatted_det)
    
    # Sort by confidence and prioritize non-empty subobjects
    formatted_detections.sort(key=lambda x: (len(x['subobjects']) == 0, -x['confidence']))
    
    # Remove duplicate detections using NMS
    filtered_detections = []
    for det in formatted_detections:
        is_duplicate = False
        for existing_det in filtered_detections:
            if existing_det['object'] == det['object']:
                is_duplicate = True
                break
        
        if not is_duplicate:
            filtered_detections.append(det)
    
    # Sort filtered detections by area for subobject processing
    for det in filtered_detections:
        bbox = det['bbox']
        det['area'] = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    
    filtered_detections.sort(key=lambda x: x['area'], reverse=True)
    
    # Process each detection
    for i, main_det in enumerate(filtered_detections):
        if main_det['id'] in used_ids:
            continue
        
        current_obj = main_det.copy()
        current_obj['subobjects'] = []
        
        # Only process if it's a potential parent object
        if main_det['object'] in subobjects_map:
            # Check all smaller objects
            for sub_det in filtered_detections[i+1:]:
                if sub_det['id'] in used_ids:
                    continue
                
                # Check if this object type can be a subobject
                valid_subobjects = [sub['name'] for sub in subobjects_map[main_det['object']]]
                if sub_det['object'] in valid_subobjects:
                    # Calculate IoU
                    iou = calculate_iou(main_det['bbox'], sub_det['bbox'])
                    min_iou = next(sub['min_iou'] 
                                 for sub in subobjects_map[main_det['object']] 
                                 if sub['name'] == sub_det['object'])
                    
                    if iou > min_iou:
                        current_obj['subobjects'].append({
                            'object': sub_det['object'],
                            'id': sub_det['id'],
                            'bbox': sub_det['bbox'],
                            'confidence': sub_det['confidence']
                        })
                        used_ids.add(sub_det['id'])
        
        processed.append(current_obj)
        used_ids.add(main_det['id'])
    
    return processed

def process_video(model, args):
    cap = open_video_capture(args.webcam, args.video)
    if not cap:
        return

    prev_time = time.time()
    frame_count = 0
    fps = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("\n\033[93m[System]\033[0m End of video or frame read error")
            break

        if frame_count % args.frame_skip != 0:
            frame_count += 1
            continue

        # Process frame
        results = model(frame)

        # Calculate FPS
        current_time = time.time()
        frame_count += 1
        if current_time - prev_time >= 1.0:
            fps = frame_count / (current_time - prev_time)
            frame_count = 0
            prev_time = current_time

        draw_detections(frame, results, fps)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def is_subobject(main_obj, sub_obj):
    """Check if sub_obj is within or appropriately overlapping with main_obj using enhanced logic"""
    main_box = [main_obj['xmin'], main_obj['ymin'], main_obj['xmax'], main_obj['ymax']]
    sub_box = [sub_obj['xmin'], sub_obj['ymin'], sub_obj['xmax'], sub_obj['ymax']]
    
    # Calculate areas
    sub_area = (sub_box[2] - sub_box[0]) * (sub_box[3] - sub_box[1])
    main_area = (main_box[2] - main_box[0]) * (main_box[3] - main_box[1])
    
    # Calculate intersection area
    intersection_area = calculate_intersection_area(main_box, sub_box)
    
    if intersection_area <= 0:
        return False  # No overlap
    
    # Calculate Intersection over Union (IoU)
    union_area = main_area + sub_area - intersection_area
    iou = intersection_area / union_area if union_area > 0 else 0
    
    # Object-specific rules
    if sub_obj['name'] in ['glasses', 'watch']:
        if sub_obj['name'] == 'glasses':
            # Glasses should occupy the upper 30% of the main object
            upper_threshold = main_box[1] + 0.3 * (main_box[3] - main_box[1])
            within_position = sub_box[1] <= upper_threshold
            return within_position and iou > 0.1
        elif sub_obj['name'] == 'watch':
            # Watch should be centered vertically and have sufficient overlap
            main_center_y = (main_box[1] + main_box[3]) / 2
            sub_center_y = (sub_box[1] + sub_box[3]) / 2
            vertical_diff = abs(sub_center_y - main_center_y)
            threshold = 0.2 * (main_box[3] - main_box[1])
            within_position = vertical_diff <= threshold
            return within_position and iou > 0.1
    else:
        # For other objects, require an IoU above a certain threshold
        return iou > 0.5

def calculate_intersection_area(box1, box2):
    """Calculate intersection area between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    return (x2 - x1) * (y2 - y1)

