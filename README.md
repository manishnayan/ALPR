# Automatic License Plate Recognition (ALPR) System

## Overview

This project implements an Automatic License Plate Recognition (ALPR) system using OpenCV and Yolo Object Detection techniques. It processes video streams to detect and recognize license plates, classify plate color, saving detected images and logging relevant data.

![alt text](https://github.com/manishnayan/ALPR/blob/main/ANPR_Arch.png?raw=true)

## Features

- Detects and recognizes license plates from video feeds.
- Saves cropped images of detected license plates.
- Classify the plate based on color (Yellow or White)
- Logs detected license plate numbers with timestamps and additional metadata.

## Creating Environment

- conda create --name anpr python=3.x
- conda activate anpr
- pip install opencv-python numpy pandas

## Model Files

Make sure to include the following model files in your project directory:

- ocr.weights
- ocr.cfg
- plates.weights
- plates.cfg
- ocr.txt (contains class labels for character detection)

## Usage

- Place the rtsp link of the live camera or video file you wish to process in the same directory.
- Activate your Conda environment:
`  conda activate anpr`
- Run the main script:
`  python main.py`
- The processed cropped license plate images will be saved in the cropped_plate_image directory.
- Detected license plate information will be logged in output/alpr_result.csv.

## References

- https://github.com/AlexeyAB/darknet
- https://github.com/guybartal/VideoAnalyticsOpenCV/tree/master
- https://github.com/crystal0929li/RandomForestModelTraining
