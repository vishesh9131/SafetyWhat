import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Object Detection System')
    parser.add_argument('--webcam', action='store_true', help='Use webcam for real-time detection')
    parser.add_argument('--video', type=str, default='v.mp4', help='Path to input video file (default: v.mp4)')
    parser.add_argument('--conf-threshold', type=float, default=0.3, help='Confidence threshold for detections')
    parser.add_argument('--iou-threshold', type=float, default=0.3, help='IoU threshold for NMS')
    parser.add_argument('--max-det', type=int, default=100, help='Maximum number of detections per image')
    parser.add_argument('--img-size', type=int, default=640, help='Image size for model input')
    parser.add_argument('--frame-skip', type=int, default=1, help='Frame skip for increased FPS')
    parser.add_argument('--model', type=str, default='yolov5s', choices=['yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x'], help='YOLOv5 model type (default: yolov5s)')
    return parser.parse_args() 