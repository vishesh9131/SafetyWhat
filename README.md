<div align="center">
    <h1> ASSESSMENT </h1>
    
</div>

<div align="center">
    <h1>Proposed name : Pixeleye</h1>
    
</div>


<div align="center">


<b>Assesment Given By : Safetywhat </b>

<b>Submitted By: Vishesh Yadav </b>

<b>Enrollment No: 12322114 </b>

<b>Degree: MCA hons AI & ML</b>

<b>Project Name: Pixeleye </b>


</div>
<br />

# Table of Contents

- [Installation](#installation)
  - [Environment Setup](#environment-setup)
- [Workflow of Safetywhat Assessment (Pixeleye)](#workflow-of-safetywhat-assessment-pixeleye)
  - [Execution Command](#execution-command)
  - [Main Processing](#main-processing)
- [Benchmarking Report](#benchmarking-report)
  - [Introduction](#introduction)
  - [System Architecture](#system-architecture)
  - [Inference Speed Results](#inference-speed-results)
  - [Optimization Strategies](#optimization-strategies)
- [References](#references)

# Installation
### Environment Setup

#### 1. Create a Conda Environment
Run the following commands to create a Conda environment named `safewt-asses` with Python 3.13:
```bash
conda create -n safewt-asses python=3.13 -y
conda activate safewt-asses
```

#### 2. Install Dependencies

Install the required dependencies using pip:
```bash
pip install -r requirements.txt
```

#### 3. Run the Code

```bash
python -u "main.py" --video locat/test_video.mp4
```

**Configurations:**

- `--webcam`: Use webcam for real-time detection.
- `--video`: Path to input video (default: `v.mp4`).
- `--conf-threshold`: Confidence threshold for detections (default: `0.3`).
- `--iou-threshold`: IoU threshold for NMS (default: `0.3`).
- `--max-det`: Max detections per image (default: `100`).
- `--img-size`: Input image size (default: `640`).
- `--frame-skip`: Skip frames for increased FPS (default: `1`).
- `--model`: YOLOv5 model type (`yolov5n`, `yolov5s`, `yolov5m`, `yolov5l`, `yolov5x`).
> default model is `yolov5s`

**Test Videos:** You can also download test videos from this drive_link: https://drive.google.com/drive/folders/1Dx4FlUM9ArTlSV0NUppw5rD8hBlZIPTy?usp=sharing

#### . Output you will get

- **Interface**: Displays the processed video with detection overlays in a user interface.
![workflow](/assets/inf.png)
> **Note**: This Output used yolov5s model (FPS: 13-25)

- **final_output.json**: Stores the detection results, including objects and subobjects, in JSON format for further analysis.
![workflow](/assets/term.png)

<b>Note:</b>
(After running the code you will get this output in **final_output.json**)
- **formatted_terminal**: Outputs formatted detection information to the terminal for real-time monitoring.
![workflow](/assets/json.png)


# Workflow of Safetywhat Assessment (pixeleye)
![workflow](/assets/d1.jpg)

This diagram illustrates the workflow of the PixEye object detection system, highlighting the interaction between various components and the flow of data from input to output.

## Execution Command

- **User Input**: The process begins with the user executing a command in the terminal:
  ```bash
  python -u "main.py" --video data/v3.mp4 --model yolov5n
  ```
  This command specifies the video file to be processed and the model to be used for detection.

## Main Processing

- **main.py**: Acts as the central hub, orchestrating the workflow. It initializes the model, processes video frames, and coordinates with other modules.

### Data Input

- **/data (sample_videos)**: Contains the video files to be processed. The specified video is loaded and passed to `main.py`.

### Processing Modules

- **image_retriever.py**: Handles saving cropped images of detected subobjects.
- **json_utils.py**: Formats and saves detection results in JSON format.
- **helper.py**: Sets up the environment and displays the application banner.
- **video.py**: Manages video capture, FPS calculation, and drawing detections on frames.
- **detection.py**: Initializes the YOLOv5 model and processes detections.
- **config.py**: Parses command-line arguments for configuration.
- **mapper.py**: Provides mappings for subobject detection.

## See it in action

![demo_detection](/assets/d6.gif)

For complete video visit : https://youtu.be/pHj8i_UcXSs
> green boundry says object and blue says its respective subobject.

# Benchmarking Report

## Introduction
This section presents the benchmarking results of the YOLOv5 models (YOLOv5n, YOLOv5s, and YOLOv5m) on various video files. The primary focus is on the inference speed, measured in frames per second (FPS), across different model configurations.

## System Architecture
- **Hardware**: Benchmarks were conducted on a CPU-based system with an Intel Core i7-9700K processor and 16 GB RAM.
- **Software**: Ubuntu 20.04 LTS, Python 3.8.20, PyTorch 2.4.1, YOLOv5 Version 2025-1-10.

## Inference Speed Results
| Video   | YOLOv5n FPS | YOLOv5s FPS | YOLOv5m FPS |
|---------|-------------|-------------|-------------|
| v.mp4   | 35.12       | 19.86       | 10.41       |
| v3.mp4  | 35.76       | 21.19       | 9.20        |
| v2.mp4  | 32.41       | 19.55       | 9.99        |
| v4.mp4  | 35.55       | 19.89       | 10.50       |
| v6.mp4  | 33.56       | 20.21       | 10.20       |
| v7.mp4  | 38.06       | 20.32       | 11.17       |
| v8.mp4  | 33.19       | 20.27       | 10.11       |

## Optimization Strategies
- **Model Pruning**: Reduce parameters and layers.
- **Quantization**: Convert weights to integer.
- **Batch Processing**: Process multiple frames simultaneously.
- **Hardware Acceleration**: Use GPUs or TPUs (in this project used cpu as per requirement)
- **Efficient Data Loading**: Optimize data loading and preprocessing.

# Benchmark

![benchmark](/assets/d7.png)



# References

- **YOLOv5 Documentation**: [YOLOv5 GitHub Repository](https://github.com/ultralytics/yolov5)
- **PyTorch Documentation**: [PyTorch Official Website](https://pytorch.org/docs/stable/index.html)
- **Conda Documentation**: [Conda Official Documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/index.html)
- **Matplotlib Documentation**: [Matplotlib Official Website](https://matplotlib.org/stable/contents.html)
- **Seaborn Documentation**: [Seaborn Official Website](https://seaborn.pydata.org/)
