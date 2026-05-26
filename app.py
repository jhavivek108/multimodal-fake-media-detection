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

# =========================================================
# VIDEO TO FRAMES
# =========================================================
def video_to_frames(video_path, output_folder="temp_frames"):
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
            ok = cv2.imwrite(frame_path, frame)
            if ok:
                saved += 1

        count += 1

    cap.release()
    return output_folder if saved > 0 else None

# =========================================================
# EXPLANATIONS
# =========================================================
def explain_image(prob):
    label = decide_label(prob)
    if label == "FAKE":
        return "The image exhibits visual inconsistencies in edge distribution and texture patterns."
    elif label == "SUSPICIOUS":
        return "The image shows mild irregularities in visual structure and texture."
    return "The image shows consistent visual structure and natural texture patterns."

def explain_video(prob):
    label = decide_label(prob)
    if label == "FAKE":
        return "The video contains inconsistencies across extracted frames."
    elif label == "SUSPICIOUS":
        return "Some extracted frames show irregular visual patterns."
    return "The extracted frames remain visually consistent and natural."

def explain_audio(prob):
    label = decide_label(prob)
    if label == "FAKE":
        return "The audio exhibits unnatural frequency characteristics and speech patterns."
    elif label == "SUSPICIOUS":
        return "The audio contains minor anomalies in frequency behavior."
    return "The audio shows natural frequency distribution and stable speech characteristics."

def explain_final(prob, used_modalities):
    label = decide_label(prob)
    joined = ", ".join(used_modalities)

    if label == "FAKE":
        return f"The final cross-modal assessment from {joined} indicates manipulated or synthetic content."
    elif label == "SUSPICIOUS":
        return f"The combined evidence from {joined} is borderline and therefore classified as suspicious."
    return f"The combined evidence from {joined} indicates authentic content."

# =========================================================
# PREDICTION FUNCTIONS
# =========================================================
def predict_image_file(image_path):
    image_path = clean_path(image_path)

    if not os.path.exists(image_path):
        return None

    if not is_image_file(image_path):
        return None

    img = load_image(image_path)
    if img is None:
        return None

    feat = extract_hog_features(img).reshape(1, -1)
    feat = img_scaler.transform(feat)

    probs = img_model.predict_proba(feat)[0]
    classes = img_model.classes_
    class_to_prob = {cls: prob for cls, prob in zip(classes, probs)}

    prob_real = float(class_to_prob.get(0, 0.0))
    prob_fake = float(class_to_prob.get(1, 0.0))
    label = decide_label(prob_fake)

    return {
        "modality": "Image",
        "prediction": label,
        "fake_prob": round(prob_fake * 100, 2),
        "real_prob": round(prob_real * 100, 2),
        "reason": explain_image(prob_fake),
        "fusion_prob": prob_fake
    }

def predict_video_file(video_path):
    video_path = clean_path(video_path)

    if not os.path.exists(video_path):
        return None

    if not is_video_file(video_path):
        return None

    frames_folder = video_to_frames(video_path)
    if frames_folder is None:
        return None

    probs = []

    for file in os.listdir(frames_folder):
        path = os.path.join(frames_folder, file)
        img = load_image(path)
        if img is not None:
            feat = extract_hog_features(img).reshape(1, -1)
            feat = frm_scaler.transform(feat)

            p = frm_model.predict_proba(feat)[0]
            classes = frm_model.classes_
            class_to_prob = {cls: prob for cls, prob in zip(classes, p)}
            prob_fake = float(class_to_prob.get(1, 0.0))
            probs.append(prob_fake)

    if not probs:
        return None

    avg_prob = float(np.mean(probs))
    label = decide_label(avg_prob)

    return {
        "modality": "Video",
        "prediction": label,
        "fake_prob": round(avg_prob * 100, 2),
        "real_prob": round((1 - avg_prob) * 100, 2),
        "reason": explain_video(avg_prob),
        "fusion_prob": avg_prob
    }

def predict_audio_file(audio_path):
    audio_path = clean_path(audio_path)

    if not os.path.exists(audio_path):
        return None

    if not is_audio_file(audio_path):
        return None

    feat = extract_mfcc_features(audio_path)
    if feat is None:
        return None

    feat = aud_scaler.transform(feat.reshape(1, -1))

    probs = aud_model.predict_proba(feat)[0]
    classes = aud_model.classes_
    class_to_prob = {cls: prob for cls, prob in zip(classes, probs)}

    prob_real = float(class_to_prob.get(0, 0.0))
    prob_fake = float(class_to_prob.get(1, 0.0))
    label = decide_label(prob_fake)

    return {
        "modality": "Audio",
        "prediction": label,
        "fake_prob": round(prob_fake * 100, 2),
        "real_prob": round(prob_real * 100, 2),
        "reason": explain_audio(prob_fake),
        "fusion_prob": prob_fake
    }

def save_file(file_obj):
    if not file_obj or file_obj.filename == "":
        return None
    ext = os.path.splitext(file_obj.filename)[1]
    unique_name = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(UPLOAD_FOLDER, unique_name)
    file_obj.save(save_path)
    return save_path

def build_terminal_output(results):
    lines = []

    if not results:
        lines.append("No valid input given.")
        return "\n".join(lines)

    for r in results:
        lines.append(f"\n--- {r['modality'].upper()} RESULT ---")
        lines.append(f"Prediction: {r['prediction']}")
        lines.append(f"Fake probability: {r['fake_prob']} %")
        lines.append(f"Real probability: {r['real_prob']} %")
        lines.append(f"Reason: {r['reason']}")

    lines.append("\n================ RESULT TABLE ================")
    lines.append(f"{'Modality':<10} {'Prediction':<12} {'Fake %':<10} {'Real %':<10}")
    lines.append("-" * 50)

    for r in results:
        lines.append(f"{r['modality']:<10} {r['prediction']:<12} {r['fake_prob']:<10} {r['real_prob']:<10}")

    probs_for_fusion = [r["fusion_prob"] for r in results]
    used_modalities = [r["modality"] for r in results]

    final_score = float(np.mean(probs_for_fusion))
    final_label = decide_label(final_score)

    lines.append("\n================ FINAL CROSS-MODAL RESULT ================")
    lines.append(f"Final Prediction: {final_label}")
    lines.append(f"Final Fake Probability: {round(final_score * 100, 2)} %")
    lines.append(f"Final Real Probability: {round((1 - final_score) * 100, 2)} %")
    lines.append(f"Explanation: {explain_final(final_score, used_modalities)}")

    return "\n".join(lines)

# =========================================================
# HTML
# =========================================================
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fake Detection Terminal View</title>
    <style>
        body {
            margin: 0;
            padding: 0;
            background: #0d1117;
            font-family: Consolas, "Courier New", monospace;
            color: #e6edf3;
        }

        .container {
            max-width: 1000px;
            margin: 30px auto;
            padding: 20px;
        }

        .card {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
        }

        h1 {
            margin-top: 0;
            color: #58a6ff;
            font-size: 26px;
        }

        .subtitle {
            color: #8b949e;
            margin-bottom: 20px;
        }

        .row {
            margin-bottom: 16px;
        }

        label {
            display: block;
            margin-bottom: 6px;
            color: #c9d1d9;
        }

        input[type="file"] {
            width: 100%;
            padding: 10px;
            background: #0d1117;
            color: #c9d1d9;
            border: 1px solid #30363d;
            border-radius: 6px;
        }

        button {
            background: #238636;
            color: white;
            border: none;
            padding: 12px 18px;
            border-radius: 6px;
            cursor: pointer;
            font-weight: bold;
            font-family: inherit;
        }

        button:hover {
            background: #2ea043;
        }

        .terminal {
            background: #010409;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 18px;
            white-space: pre-wrap;
            overflow-x: auto;
            line-height: 1.6;
            color: #c9d1d9;
            min-height: 220px;
        }

        .error {
            color: #ff7b72;
        }

        .ok {
            color: #3fb950;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <h1>Multimodal Fake Detection System</h1>
            <div class="subtitle">VS Code terminal-style output in browser</div>

            <form method="POST" enctype="multipart/form-data">
                <div class="row">
                    <label>Upload Image</label>
                    <input type="file" name="image" accept=".jpg,.jpeg,.png,.bmp,.webp">
                </div>

                <div class="row">
                    <label>Upload Video</label>
                    <input type="file" name="video" accept=".mp4,.avi,.mov,.mkv">
                </div>

                <div class="row">
                    <label>Upload Audio</label>
                    <input type="file" name="audio" accept=".wav,.mp3,.m4a,.flac,.ogg">
                </div>

                <button type="submit">Run Detection</button>
            </form>
        </div>

        <div class="card">
            <div class="terminal">{% if output %}{{ output }}{% else %}Ready. Upload file(s) and click Run Detection.{% endif %}</div>
        </div>
    </div>
</body>
</html>
"""

# =========================================================
# ROUTES
# =========================================================
@app.route("/favicon.ico")
def favicon():
    return "", 204

@app.route("/", methods=["GET", "POST"])
def index():
    output = ""

    if request.method == "POST":
        results = []

        image_file = request.files.get("image")
        video_file = request.files.get("video")
        audio_file = request.files.get("audio")

        image_path = save_file(image_file) if image_file and image_file.filename else None
        video_path = save_file(video_file) if video_file and video_file.filename else None
        audio_path = save_file(audio_file) if audio_file and audio_file.filename else None

        if image_path:
            r = predict_image_file(image_path)
            if r:
                results.append(r)

        if video_path:
            r = predict_video_file(video_path)
            if r:
                results.append(r)

        if audio_path:
            r = predict_audio_file(audio_path)
            if r:
                results.append(r)

        output = build_terminal_output(results)

    return render_template_string(HTML, output=output)

# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    app.run(debug=True)