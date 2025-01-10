import cv2
import time

def open_video_capture(use_webcam, video_path):
    cap = cv2.VideoCapture(1 if use_webcam else video_path)
    if not cap.isOpened():
        print("\n\033[91m╭── Error ──────────────────────────────╮")
        print("│ Could not open video source.")
        print("│ Please check the video file or webcam.")
        print("╰────────────────────────────────────────╯\033[0m")
        return None
    return cap

def calculate_fps(frame_count, prev_time):
    current_time = time.time()
    frame_count += 1
    fps = 0  # Initialize fps with a default value
    if current_time - prev_time >= 1.0:
        fps = frame_count
        frame_count = 0
        prev_time = current_time
    return frame_count, fps

def draw_detections(frame, results, fps):
    for *xyxy, conf, cls in results.xyxy[0]:
        label = f"{results.names[int(cls)]} {conf:.2f}"
        cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
        cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.imshow('SAFETYWHAT ASSESSMENT', frame) 