# Multimodal Fake Media Detection System

A machine learning-based system for detecting fake or manipulated media using image, video-frame, and audio analysis.

## Features

* Image fake detection using HOG features
* Video frame authenticity analysis
* Audio fake detection using MFCC features
* Multimodal fusion-based prediction
* Flask web interface
* Real/Fake/Suspicious classification

---

## Tech Stack

* Python
* Flask
* OpenCV
* Librosa
* Scikit-learn
* XGBoost
* NumPy

---

## Supported Formats

| Type  | Formats        |
| ----- | -------------- |
| Image | JPG, PNG, WEBP |
| Video | MP4, AVI, MOV  |
| Audio | WAV, MP3, FLAC |

---

## ML Pipeline

* HOG feature extraction for images/videos
* MFCC extraction for audio
* SVM/XGBoost classification
* Cross-modal probability fusion