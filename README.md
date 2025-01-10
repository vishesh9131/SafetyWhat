# PixEye: SAFETYWHAT ASSESSMENT

## Overview

PixEye is an object detection system designed to process video inputs and identify objects and their subobjects in real-time. It leverages the YOLOv5 model for detection and provides functionalities to save detection results and subobject images, calculate Intersection over Union (IoU), and manage video processing efficiently.

## Features

- **Real-time Object Detection**: Utilizes YOLOv5 for detecting objects in video frames.
- **Subobject Detection**: Identifies subobjects within detected objects using a predefined mapping.
- **JSON Output**: Saves detection results in JSON format for further analysis.
- **Image Saving**: Crops and saves images of detected subobjects.
- **FPS Calculation**: Calculates and displays frames per second (FPS) for performance monitoring.
- **Video Source Flexibility**: Supports both webcam and video file inputs.

## Components

### 1. Main Script (`main.py`)

- **Initialization**: Sets up the environment and initializes the detection model.
- **Video Processing**: Captures video frames, processes them for object detection, and handles frame skipping for performance.
- **Detection Handling**: Converts detection results into a structured format, processes subobjects, and saves results.
- **Display**: Draws detection results and FPS on video frames and displays them.

### 2. Helper Functions (`helper.py`)

- **Environment Setup**: Configures the environment to suppress warnings and optimize performance.
- **Banner Display**: Clears the console and displays a banner for the application.

### 3. Video Utilities (`video.py`)

- **Video Capture**: Opens video sources and handles errors if the source is unavailable.
- **FPS Calculation**: Provides a utility to calculate FPS based on frame count and time.
- **Drawing Utilities**: Draws detection results and FPS on video frames.

### 4. JSON Utilities (`json_utils.py`)

- **Detection Formatting**: Formats detection results into JSON-compatible structures.
- **Information Display**: Prints detection information, including subobjects, to the console.
- **File Management**: Merges individual JSON files into a single output and cleans up temporary files.

### 5. Configuration (`config.py`)

- **Argument Parsing**: Defines and parses command-line arguments for configuring the detection system, such as video source, confidence threshold, and model type.

### 6. Detection Logic (`detection.py`)

- **Model Initialization**: Loads and configures the YOLOv5 model with specified parameters.
- **Detection Processing**: Processes detections to identify subobjects and remove duplicates.
- **Video Processing**: Manages the main loop for processing video frames and handling user input.

### 7. Subobject Mapping (`mapper.py`)

- **Subobject Definitions**: Provides a mapping of main objects to potential subobjects with minimum IoU thresholds for valid detection.

### 8. Image Retrieval (`image_retriever.py`)

- **Image Saving**: Saves cropped images of detected subobjects to a specified directory.

## Usage

1. **Setup**: Ensure all dependencies are installed, including OpenCV and PyTorch.
2. **Run the Application**: Use the command line to start the application with desired parameters:
   ```bash
   python main.py --video path/to/video.mp4 --conf-threshold 0.3 --iou-threshold 0.3
   ```
3. **View Results**: The application will display the video with detection overlays and save results in JSON format.

## Code References

- Main script and detection logic: `main.py` (startLine: 16, endLine: 220)
- Helper functions: `helper.py` (startLine: 5, endLine: 21)
- Video utilities: `video.py` (startLine: 4, endLine: 30)
- JSON utilities: `json_utils.py` (startLine: 14, endLine: 103)
- Configuration: `config.py` (startLine: 3, endLine: 13)
- Detection logic: `detection.py` (startLine: 20, endLine: 188)
- Subobject mapping: `mapper.py` (startLine: 1, endLine: 29)
- Image retrieval: `image_retriever.py` (startLine: 5, endLine: 27)

## Conclusion

PixEye provides a comprehensive solution for real-time object detection and analysis, with a focus on flexibility and performance. By leveraging advanced detection models and efficient processing techniques, it offers a robust platform for various applications in safety and surveillance.
