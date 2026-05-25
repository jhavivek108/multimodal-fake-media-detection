import os
import shutil
import uuid
import warnings

from flask import Flask, request, render_template_string
import cv2
import librosa
import numpy as np
import joblib
from skimage.feature import hog

warnings.filterwarnings("ignore")

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =========================================================
# LOAD TRAINED MODELS
# =========================================================
img_model = joblib.load("saved_models/img_model.pkl")
img_scaler = joblib.load("saved_models/img_scaler.pkl")

frm_model = joblib.load("saved_models/frm_model.pkl")
frm_scaler = joblib.load("saved_models/frm_scaler.pkl")

aud_model = joblib.load("saved_models/aud_model.pkl")
aud_scaler = joblib.load("saved_models/aud_scaler.pkl")

# =========================================================
# SETTINGS
# =========================================================
IMG_SIZE = (128, 128)
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")
AUDIO_EXTS = (".wav", ".mp3", ".m4a", ".flac", ".ogg")

THRESHOLD_FAKE = 0.28
THRESHOLD_SUSPICIOUS = 0.18

# =========================================================
# HELPERS
# =========================================================
def clean_path(p):
    return p.strip().replace('"', '').replace("'", "")

def is_image_file(path):
    return path.lower().endswith(IMAGE_EXTS)

def is_video_file(path):
    return path.lower().endswith(VIDEO_EXTS)

def is_audio_file(path):
    return path.lower().endswith(AUDIO_EXTS)

def decide_label(prob):
    if prob >= THRESHOLD_FAKE:
        return "FAKE"
    elif prob >= THRESHOLD_SUSPICIOUS:
        return "SUSPICIOUS"
    return "REAL"

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.resize(img, IMG_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray / 255.0
    return gray

def extract_hog_features(gray_img):
    return hog(
        gray_img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

def extract_mfcc_features(audio_path, max_len=300):
    try:
        audio, sr = librosa.load(audio_path, sr=22050)
        if audio is None or len(audio) == 0:
            return None

        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=30)
        mfcc = mfcc.flatten()

        if len(mfcc) < max_len:
            mfcc = np.pad(mfcc, (0, max_len - len(mfcc)))
        else:
            mfcc = mfcc[:max_len]

        return mfcc
    except Exception:
        return None