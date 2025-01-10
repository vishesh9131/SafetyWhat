import cv2
import time
import os
from detection import initialize_model
from config import get_args

def benchmark_inference(video_path, model_type, results_file):
    args = get_args()
    model = initialize_model(args.conf_threshold, args.iou_threshold, args.max_det, args.img_size, model_type)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_path}.")
        return

    total_frames = 0
    total_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()
        
        # Perform detection
        results = model(frame)
        
        end_time = time.time()
        total_time += (end_time - start_time)
        total_frames += 1

    cap.release()
    avg_fps = total_frames / total_time if total_time > 0 else 0
    result = (
        f"Processed {total_frames} frames in {total_time:.2f} seconds.\n"
        f"Average FPS: {avg_fps:.2f}\n"
        f"Model: {model_type}\n"
        f"Video: {video_path}\n"
        f"Confidence Threshold: {args.conf_threshold}\n"
        f"IoU Threshold: {args.iou_threshold}\n"
        f"Max Detections: {args.max_det}\n"
        f"Image Size: {args.img_size}\n"
        f"Frame Skip: {args.frame_skip}\n"
        + "-" * 50 + "\n"
    )
    print(result)
    with open(results_file, 'a') as f:
        f.write(result)

if __name__ == "__main__":
    video_folder = "data"
    models = ['yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']
    results_file = "benchmark_results.txt"

    # Clear previous results
    open(results_file, 'w').close()

    for model_type in models:
        for video_file in os.listdir(video_folder):
            if video_file.endswith(".mp4"):
                video_path = os.path.join(video_folder, video_file)
                benchmark_inference(video_path, model_type, results_file)
