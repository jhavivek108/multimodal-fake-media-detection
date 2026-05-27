import os
import io
import uuid
import base64
import shutil
import warnings

from flask import Flask, request, render_template_string
import cv2
import librosa
import librosa.display
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.feature import hog

warnings.filterwarnings("ignore")

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
TEMP_FRAMES = "temp_frames"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEMP_FRAMES, exist_ok=True)

# =========================================================
# LOAD MODELS
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
def decide_label(prob):
    if prob >= THRESHOLD_FAKE:
        return "FAKE"
    elif prob >= THRESHOLD_SUSPICIOUS:
        return "SUSPICIOUS"
    return "REAL"

def get_extension(path):
    return os.path.splitext(path)[1].lower()

def detect_media_type(path):
    ext = get_extension(path)
    if ext in IMAGE_EXTS:
        return "Image"
    if ext in VIDEO_EXTS:
        return "Video"
    if ext in AUDIO_EXTS:
        return "Audio"
    return "Unknown"

def save_file(file_obj):
    if not file_obj or file_obj.filename == "":
        return None
    ext = os.path.splitext(file_obj.filename)[1]
    name = f"{uuid.uuid4().hex}{ext}"
    path = os.path.join(UPLOAD_FOLDER, name)
    file_obj.save(path)
    return path

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.resize(img, IMG_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray / 255.0

def extract_hog_features(gray_img):
    return hog(
        gray_img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

def extract_hog_with_visual(gray_img):
    features, hog_image = hog(
        gray_img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=True
    )
    return features, hog_image

def extract_mfcc_features(audio_path, max_len=300):
    try:
        audio, sr = librosa.load(audio_path, sr=22050)
        if audio is None or len(audio) == 0:
            return None, None, None

        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=30)
        flat = mfcc.flatten()

        if len(flat) < max_len:
            flat = np.pad(flat, (0, max_len - len(flat)))
        else:
            flat = flat[:max_len]

        return flat, audio, sr
    except Exception:
        return None, None, None

def video_to_frames(video_path, output_folder):
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    count = 0
    saved = 0
    frame_step = 10

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_step == 0:
            frame_path = os.path.join(output_folder, f"frame_{saved}.jpg")
            if cv2.imwrite(frame_path, frame):
                saved += 1

        count += 1

    cap.release()
    return output_folder if saved > 0 else None

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=160)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_b64

def image_file_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def make_hog_overlay_base64(image_path):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None

    img_bgr = cv2.resize(img_bgr, IMG_SIZE)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    _, hog_vis = extract_hog_with_visual(gray / 255.0)
    hog_norm = cv2.normalize(hog_vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    sharp_map = cv2.Laplacian(gray, cv2.CV_64F)
    sharp_map = np.absolute(sharp_map)
    sharp_map = cv2.normalize(sharp_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    overlay = img_bgr.copy()

    green_mask = np.zeros_like(img_bgr)
    green_mask[:, :, 1] = hog_norm

    red_mask = np.zeros_like(img_bgr)
    red_mask[:, :, 2] = sharp_map

    overlay = cv2.addWeighted(overlay, 0.70, green_mask, 0.70, 0)
    overlay = cv2.addWeighted(overlay, 0.85, red_mask, 0.45, 0)

    ok, enc = cv2.imencode(".png", overlay)
    if not ok:
        return None

    return base64.b64encode(enc.tobytes()).decode("utf-8")

def make_audio_waveform_base64(audio, sr, title):
    fig, ax = plt.subplots(figsize=(6, 3))
    librosa.display.waveshow(audio, sr=sr, ax=ax, color="#22c55e")
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    fig.tight_layout()
    return fig_to_base64(fig)

def make_mfcc_graph_base64(audio, sr, title):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=30)
    fig, ax = plt.subplots(figsize=(6, 3))
    img = librosa.display.specshow(mfcc, x_axis="time", sr=sr, ax=ax, cmap="magma")
    ax.set_title(title)
    fig.colorbar(img, ax=ax)
    fig.tight_layout()
    return fig_to_base64(fig)

# =========================================================
# IMPORTANT FEATURE SUMMARIES
# =========================================================
def summarize_hog_features(hog_feat, gray_img):
    sharp_map = cv2.Laplacian((gray_img * 255).astype(np.uint8), cv2.CV_64F)
    sharpness_score = float(np.mean(np.abs(sharp_map)))

    return {
        "Mean Gradient Strength": round(float(np.mean(hog_feat)), 4),
        "Max Gradient Strength": round(float(np.max(hog_feat)), 4),
        "Edge Density": round(float(np.sum(hog_feat > np.mean(hog_feat)) / len(hog_feat)), 4),
        "Texture Variation": round(float(np.std(hog_feat)), 4),
        "Sharpness Score": round(sharpness_score, 4),
    }

def summarize_video_features(avg_feat, sample_img):
    sharp_map = cv2.Laplacian((sample_img * 255).astype(np.uint8), cv2.CV_64F)
    sharpness_score = float(np.mean(np.abs(sharp_map)))

    return {
        "Mean Frame Gradient": round(float(np.mean(avg_feat)), 4),
        "Max Frame Gradient": round(float(np.max(avg_feat)), 4),
        "Frame Edge Density": round(float(np.sum(avg_feat > np.mean(avg_feat)) / len(avg_feat)), 4),
        "Frame Texture Variation": round(float(np.std(avg_feat)), 4),
        "Frame Sharpness Score": round(sharpness_score, 4),
    }

def summarize_audio_features(mfcc_feat, audio):
    signal_energy = float(np.mean(audio ** 2))
    zero_crossing_rate = float(np.mean(librosa.feature.zero_crossing_rate(y=audio)))

    return {
        "MFCC Mean": round(float(np.mean(mfcc_feat)), 4),
        "MFCC Std Dev": round(float(np.std(mfcc_feat)), 4),
        "MFCC Max": round(float(np.max(mfcc_feat)), 4),
        "Signal Energy": round(signal_energy, 4),
        "Zero Crossing Rate": round(zero_crossing_rate, 4),
    }

# =========================================================
# FEATURE BASIS TEXT
# =========================================================
FEATURE_BASIS = {
    "HOG": "Important displayed features are derived from HOG-based edge orientation, contrast variation, texture behavior, and sharpness. These are the most meaningful summarized features shown in the interface.",
    "Frame-HOG": "Important displayed features are derived from averaged HOG patterns across sampled video frames, highlighting frame gradients, edge density, texture variation, and sharpness.",
    "MFCC": "Important displayed features are derived from MFCC-based spectral behavior, including cepstral statistics, signal energy, and zero-crossing rate."
}

# =========================================================
# PREDICTION FUNCTIONS
# =========================================================
def predict_image_file(file_path):
    img = load_image(file_path)
    if img is None:
        return None

    hog_feat = extract_hog_features(img)
    feat_scaled = img_scaler.transform(hog_feat.reshape(1, -1))

    probs = img_model.predict_proba(feat_scaled)[0]
    classes = img_model.classes_
    class_to_prob = {cls: prob for cls, prob in zip(classes, probs)}

    prob_fake = float(class_to_prob.get(1, 0.0))
    prob_real = float(class_to_prob.get(0, 0.0))

    return {
        "media_type": "Image",
        "feature_name": "HOG",
        "feature_count": len(hog_feat),
        "feature_summary": summarize_hog_features(hog_feat, img),
        "prediction": decide_label(prob_fake),
        "fake_prob": round(prob_fake * 100, 2),
        "real_prob": round(prob_real * 100, 2),
        "fusion_prob": prob_fake,
        "basis": FEATURE_BASIS["HOG"],
        "original_preview": image_file_to_base64(file_path),
        "feature_preview_image": make_hog_overlay_base64(file_path),
        "graph_preview": None,
        "mfcc_graph": None,
    }

def predict_video_file(file_path):
    unique_folder = os.path.join(TEMP_FRAMES, uuid.uuid4().hex)
    frames_folder = video_to_frames(file_path, unique_folder)
    if frames_folder is None:
        return None

    all_probs = []
    all_features = []
    first_frame = None
    first_frame_hog_overlay = None
    sample_img_for_summary = None

    for fname in os.listdir(frames_folder):
        path = os.path.join(frames_folder, fname)
        img = load_image(path)
        if img is not None:
            if first_frame is None:
                first_frame = image_file_to_base64(path)
                first_frame_hog_overlay = make_hog_overlay_base64(path)

            if sample_img_for_summary is None:
                sample_img_for_summary = img

            hog_feat = extract_hog_features(img)
            all_features.append(hog_feat)

            feat_scaled = frm_scaler.transform(hog_feat.reshape(1, -1))
            probs = frm_model.predict_proba(feat_scaled)[0]
            classes = frm_model.classes_
            class_to_prob = {cls: prob for cls, prob in zip(classes, probs)}
            all_probs.append(float(class_to_prob.get(1, 0.0)))

    if not all_probs or not all_features or sample_img_for_summary is None:
        return None

    avg_prob = float(np.mean(all_probs))
    avg_feat = np.mean(np.array(all_features), axis=0)

    return {
        "media_type": "Video",
        "feature_name": "Frame-HOG",
        "feature_count": len(avg_feat),
        "feature_summary": summarize_video_features(avg_feat, sample_img_for_summary),
        "prediction": decide_label(avg_prob),
        "fake_prob": round(avg_prob * 100, 2),
        "real_prob": round((1 - avg_prob) * 100, 2),
        "fusion_prob": avg_prob,
        "basis": FEATURE_BASIS["Frame-HOG"],
        "original_preview": first_frame,
        "feature_preview_image": first_frame_hog_overlay,
        "graph_preview": None,
        "mfcc_graph": None,
    }

def predict_audio_file(file_path):
    mfcc_feat, audio, sr = extract_mfcc_features(file_path)
    if mfcc_feat is None:
        return None

    feat_scaled = aud_scaler.transform(mfcc_feat.reshape(1, -1))
    probs = aud_model.predict_proba(feat_scaled)[0]
    classes = aud_model.classes_
    class_to_prob = {cls: prob for cls, prob in zip(classes, probs)}

    prob_fake = float(class_to_prob.get(1, 0.0))
    prob_real = float(class_to_prob.get(0, 0.0))

    return {
        "media_type": "Audio",
        "feature_name": "MFCC",
        "feature_count": len(mfcc_feat),
        "feature_summary": summarize_audio_features(mfcc_feat, audio),
        "prediction": decide_label(prob_fake),
        "fake_prob": round(prob_fake * 100, 2),
        "real_prob": round(prob_real * 100, 2),
        "fusion_prob": prob_fake,
        "basis": FEATURE_BASIS["MFCC"],
        "original_preview": None,
        "feature_preview_image": None,
        "graph_preview": make_audio_waveform_base64(audio, sr, "Audio Waveform"),
        "mfcc_graph": make_mfcc_graph_base64(audio, sr, "MFCC Feature Map"),
    }

def analyze_single_file(file_path, display_name):
    media_type = detect_media_type(file_path)

    if media_type == "Image":
        result = predict_image_file(file_path)
    elif media_type == "Video":
        result = predict_video_file(file_path)
    elif media_type == "Audio":
        result = predict_audio_file(file_path)
    else:
        result = None

    if result is None:
        return {
            "name": display_name,
            "media_type": media_type if media_type != "Unknown" else "Unsupported",
            "feature_name": "-",
            "feature_count": 0,
            "feature_summary": {},
            "prediction": "INVALID",
            "fake_prob": 0,
            "real_prob": 0,
            "fusion_prob": 0,
            "basis": "Unsupported or invalid input.",
            "original_preview": None,
            "feature_preview_image": None,
            "graph_preview": None,
            "mfcc_graph": None,
        }

    result["name"] = display_name
    return result

def analyze_group(group_name, files_dict):
    results = []

    for media_key, file_path in files_dict.items():
        if file_path:
            pretty_name = f"{group_name} {media_key.title()}"
            results.append(analyze_single_file(file_path, pretty_name))

    probs = [r["fusion_prob"] for r in results if r["prediction"] != "INVALID"]
    if probs:
        final_score = float(np.mean(probs))
        final_result = {
            "group_name": group_name,
            "prediction": decide_label(final_score),
            "fake_prob": round(final_score * 100, 2),
            "real_prob": round((1 - final_score) * 100, 2),
        }
    else:
        final_result = {
            "group_name": group_name,
            "prediction": "INVALID",
            "fake_prob": 0,
            "real_prob": 0,
        }

    return results, final_result

# =========================================================
# HTML
# =========================================================
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multimodal Fake Detection System</title>
    <style>
        body {
            background: #0f172a;
            color: #e5e7eb;
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 24px;
        }
        .container {
            max-width: 1450px;
            margin: auto;
        }
        .card {
            background: #111827;
            border-radius: 14px;
            padding: 22px;
            margin-bottom: 20px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.25);
        }
        .grid2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 18px;
        }
        .upload-box, .preview-box {
            background: #1f2937;
            padding: 16px;
            border-radius: 12px;
        }
        h1, h2, h3 {
            margin-top: 0;
        }
        .sub {
            color: #9ca3af;
            margin-bottom: 18px;
        }
        label {
            display: block;
            margin: 10px 0 6px;
            font-weight: bold;
        }
        input[type="file"] {
            width: 100%;
            padding: 10px;
            background: #0b1220;
            color: #e5e7eb;
            border: 1px solid #374151;
            border-radius: 8px;
        }
        button {
            background: #2563eb;
            color: white;
            border: none;
            padding: 12px 18px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 15px;
            font-weight: bold;
            margin-top: 18px;
        }
        button:hover {
            background: #1d4ed8;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 14px;
        }
        th, td {
            border: 1px solid #374151;
            padding: 10px;
            text-align: center;
            vertical-align: top;
        }
        th {
            background: #1f2937;
        }
        td {
            background: #0b1220;
        }
        .tag {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: bold;
        }
        .real { background: #14532d; color: #bbf7d0; }
        .fake { background: #7f1d1d; color: #fecaca; }
        .sus { background: #78350f; color: #fde68a; }
        .invalid { background: #374151; color: #e5e7eb; }
        .mini-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 14px;
        }
        .imgbox {
            background: #0b1220;
            border: 1px solid #374151;
            border-radius: 10px;
            padding: 12px;
            margin-top: 14px;
        }
        .imgbox img {
            width: 100%;
            max-width: 340px;
            border-radius: 8px;
            border: 1px solid #374151;
            display: block;
            margin-top: 8px;
        }
        .feature-text {
            background: #0b1220;
            border: 1px solid #374151;
            border-radius: 10px;
            padding: 12px;
            margin-top: 14px;
            line-height: 1.6;
            white-space: pre-wrap;
        }
        .legend {
            font-size: 13px;
            margin-top: 8px;
            line-height: 1.5;
            color: #cbd5e1;
        }
        .error {
            background: #7f1d1d;
            color: #fecaca;
            padding: 12px;
            border-radius: 10px;
        }
        .section-gap {
            margin-top: 22px;
        }
        .small-note {
            color: #cbd5e1;
            font-size: 14px;
            margin-top: 8px;
        }
    </style>
</head>
<body>
<div class="container">
    <div class="card">
        <h1>Multimodal Fake Detection System</h1>
        <div class="sub">
            Upload image, video, and audio for A and B. The interface shows only important summarized features, while the complete feature vector is still used internally for prediction.
        </div>

        <form method="POST" enctype="multipart/form-data">
            <div class="grid2">
                <div class="upload-box">
                    <h3>Input Group A</h3>
                    <label>Image A</label>
                    <input type="file" name="image_a" accept=".jpg,.jpeg,.png,.bmp,.webp">
                    <label>Video A</label>
                    <input type="file" name="video_a" accept=".mp4,.avi,.mov,.mkv">
                    <label>Audio A</label>
                    <input type="file" name="audio_a" accept=".wav,.mp3,.m4a,.flac,.ogg">
                </div>

                <div class="upload-box">
                    <h3>Input Group B</h3>
                    <label>Image B</label>
                    <input type="file" name="image_b" accept=".jpg,.jpeg,.png,.bmp,.webp">
                    <label>Video B</label>
                    <input type="file" name="video_b" accept=".mp4,.avi,.mov,.mkv">
                    <label>Audio B</label>
                    <input type="file" name="audio_b" accept=".wav,.mp3,.m4a,.flac,.ogg">
                </div>
            </div>
            <button type="submit">Run Comparison</button>
        </form>
    </div>

    {% if error %}
    <div class="card"><div class="error">{{ error }}</div></div>
    {% endif %}

    {% if results_a or results_b %}
    <div class="card">
        <h2>ML Output Table</h2>
        <table>
            <tr>
                <th>Input</th>
                <th>Media Type</th>
                <th>Feature Type</th>
                <th>Total Features</th>
                <th>Prediction</th>
                <th>Fake %</th>
                <th>Real %</th>
            </tr>
            {% for r in results_a + results_b %}
            <tr>
                <td>{{ r.name }}</td>
                <td>{{ r.media_type }}</td>
                <td>{{ r.feature_name }}</td>
                <td>{{ r.feature_count }}</td>
                <td>
                    <span class="tag {% if r.prediction == 'REAL' %}real{% elif r.prediction == 'FAKE' %}fake{% elif r.prediction == 'SUSPICIOUS' %}sus{% else %}invalid{% endif %}">
                        {{ r.prediction }}
                    </span>
                </td>
                <td>{{ r.fake_prob }}</td>
                <td>{{ r.real_prob }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>

    <div class="card">
        <h2>Important Feature Summary</h2>
        <div class="grid2">
            <div class="preview-box">
                <h3>Group A</h3>
                {% for r in results_a %}
                <div class="feature-text">
<b>{{ r.name }}</b>
Media Type: {{ r.media_type }}
Feature Type: {{ r.feature_name }}
Basis: {{ r.basis }}
Only important summarized features are displayed below. Full feature vector is used internally by the model.
                </div>

                {% if r.feature_summary %}
                <table>
                    <tr>
                        <th>Important Feature</th>
                        <th>Value</th>
                    </tr>
                    {% for key, value in r.feature_summary.items() %}
                    <tr>
                        <td>{{ key }}</td>
                        <td>{{ value }}</td>
                    </tr>
                    {% endfor %}
                    <tr>
                        <td><b>Total Feature Count</b></td>
                        <td><b>{{ r.feature_count }}</b></td>
                    </tr>
                </table>
                <div class="small-note">
                    These are the important displayed features for interpretation.
                </div>
                {% endif %}

                <div class="mini-grid">
                    {% if r.original_preview %}
                    <div class="imgbox">
                        <b>Uploaded Preview</b>
                        <img src="data:image/png;base64,{{ r.original_preview }}">
                    </div>
                    {% endif %}

                    {% if r.feature_preview_image %}
                    <div class="imgbox">
                        <b>Feature Overlay</b>
                        <img src="data:image/png;base64,{{ r.feature_preview_image }}">
                        <div class="legend">
                            Green = HOG edge / gradient features<br>
                            Red = sharpness / strong intensity change
                        </div>
                    </div>
                    {% endif %}
                </div>

                {% if r.graph_preview %}
                <div class="imgbox">
                    <b>Audio Waveform</b>
                    <img src="data:image/png;base64,{{ r.graph_preview }}">
                </div>
                {% endif %}

                {% if r.mfcc_graph %}
                <div class="imgbox">
                    <b>MFCC Feature Map</b>
                    <img src="data:image/png;base64,{{ r.mfcc_graph }}">
                </div>
                {% endif %}

                <div class="section-gap"></div>
                {% endfor %}
            </div>

            <div class="preview-box">
                <h3>Group B</h3>
                {% for r in results_b %}
                <div class="feature-text">
<b>{{ r.name }}</b>
Media Type: {{ r.media_type }}
Feature Type: {{ r.feature_name }}
Basis: {{ r.basis }}
Only important summarized features are displayed below. Full feature vector is used internally by the model.
                </div>

                {% if r.feature_summary %}
                <table>
                    <tr>
                        <th>Important Feature</th>
                        <th>Value</th>
                    </tr>
                    {% for key, value in r.feature_summary.items() %}
                    <tr>
                        <td>{{ key }}</td>
                        <td>{{ value }}</td>
                    </tr>
                    {% endfor %}
                    <tr>
                        <td><b>Total Feature Count</b></td>
                        <td><b>{{ r.feature_count }}</b></td>
                    </tr>
                </table>
                <div class="small-note">
                    These are the important displayed features for interpretation.
                </div>
                {% endif %}

                <div class="mini-grid">
                    {% if r.original_preview %}
                    <div class="imgbox">
                        <b>Uploaded Preview</b>
                        <img src="data:image/png;base64,{{ r.original_preview }}">
                    </div>
                    {% endif %}

                    {% if r.feature_preview_image %}
                    <div class="imgbox">
                        <b>Feature Overlay</b>
                        <img src="data:image/png;base64,{{ r.feature_preview_image }}">
                        <div class="legend">
                            Green = HOG edge / gradient features<br>
                            Red = sharpness / strong intensity change
                        </div>
                    </div>
                    {% endif %}
                </div>

                {% if r.graph_preview %}
                <div class="imgbox">
                    <b>Audio Waveform</b>
                    <img src="data:image/png;base64,{{ r.graph_preview }}">
                </div>
                {% endif %}

                {% if r.mfcc_graph %}
                <div class="imgbox">
                    <b>MFCC Feature Map</b>
                    <img src="data:image/png;base64,{{ r.mfcc_graph }}">
                </div>
                {% endif %}

                <div class="section-gap"></div>
                {% endfor %}
            </div>
        </div>
    </div>

    <div class="card">
        <h2>Final Comparison Table</h2>
        <table>
            <tr>
                <th>Parameter</th>
                <th>Group A</th>
                <th>Group B</th>
            </tr>
            <tr>
                <td>Total Uploaded Modalities</td>
                <td>{{ results_a|length }}</td>
                <td>{{ results_b|length }}</td>
            </tr>
            <tr>
                <td>Final Prediction</td>
                <td>{{ final_a.prediction }}</td>
                <td>{{ final_b.prediction }}</td>
            </tr>
            <tr>
                <td>Fake Probability</td>
                <td>{{ final_a.fake_prob }}%</td>
                <td>{{ final_b.fake_prob }}%</td>
            </tr>
            <tr>
                <td>Real Probability</td>
                <td>{{ final_a.real_prob }}%</td>
                <td>{{ final_b.real_prob }}%</td>
            </tr>
        </table>
    </div>
    {% endif %}
</div>
</body>
</html>
"""

# =========================================================
# ROUTE
# =========================================================
@app.route("/", methods=["GET", "POST"])
def index():
    results_a = []
    results_b = []
    final_a = None
    final_b = None
    error = None

    if request.method == "POST":
        files_a = {
            "image": save_file(request.files.get("image_a")) if request.files.get("image_a") and request.files.get("image_a").filename else None,
            "video": save_file(request.files.get("video_a")) if request.files.get("video_a") and request.files.get("video_a").filename else None,
            "audio": save_file(request.files.get("audio_a")) if request.files.get("audio_a") and request.files.get("audio_a").filename else None,
        }

        files_b = {
            "image": save_file(request.files.get("image_b")) if request.files.get("image_b") and request.files.get("image_b").filename else None,
            "video": save_file(request.files.get("video_b")) if request.files.get("video_b") and request.files.get("video_b").filename else None,
            "audio": save_file(request.files.get("audio_b")) if request.files.get("audio_b") and request.files.get("audio_b").filename else None,
        }

        if not any(files_a.values()) and not any(files_b.values()):
            error = "Please upload at least one file in A or B."
        else:
            results_a, final_a = analyze_group("A", files_a)
            results_b, final_b = analyze_group("B", files_b)

    return render_template_string(
        HTML,
        results_a=results_a,
        results_b=results_b,
        final_a=final_a,
        final_b=final_b,
        error=error
    )

# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    app.run(debug=True, port=5001)