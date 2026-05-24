import os
import shutil
import cv2
import librosa
import numpy as np
import joblib
from skimage.feature import hog

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
        return "Edge and texture patterns look unnatural."
    elif label == "SUSPICIOUS":
        return "Some visual patterns look unusual."
    return "Edge and texture patterns look natural."

def explain_video(prob):
    label = decide_label(prob)
    if label == "FAKE":
        return "Several extracted frames look inconsistent."
    elif label == "SUSPICIOUS":
        return "Some extracted frames look unusual."
    return "Most extracted frames look stable and natural."

def explain_audio(prob):
    label = decide_label(prob)
    if label == "FAKE":
        return "Speech-frequency pattern looks synthetic."
    elif label == "SUSPICIOUS":
        return "Some frequency patterns look unusual."
    return "Speech-frequency pattern looks natural."

# =========================================================
# PREDICTION FUNCTIONS
# =========================================================
def predict_image_file(image_path):
    image_path = clean_path(image_path)

    if not os.path.exists(image_path):
        print("\n--- IMAGE RESULT ---")
        print("Input is invalid or path not found.")
        return None

    if not is_image_file(image_path):
        print("\n--- IMAGE RESULT ---")
        print("Input is invalid or corrupted.")
        return None

    img = load_image(image_path)
    if img is None:
        print("\n--- IMAGE RESULT ---")
        print("Input image is invalid or corrupted.")
        return None

    feat = extract_hog_features(img).reshape(1, -1)
    feat = img_scaler.transform(feat)

    probs = img_model.predict_proba(feat)[0]
    classes = img_model.classes_
    class_to_prob = {cls: prob for cls, prob in zip(classes, probs)}

    prob_real = class_to_prob.get(0, 0.0)
    prob_fake = class_to_prob.get(1, 0.0)
    label = decide_label(prob_fake)

    print("\n--- IMAGE DEBUG ---")
    print("Testing image:", image_path)
    print("Classes:", classes)
    print("DEBUG fake prob:", prob_fake)
    print("DEBUG real prob:", prob_real)

    print("\n--- IMAGE RESULT ---")
    print("Prediction:", label)
    print("Fake probability:", round(prob_fake * 100, 2), "%")
    print("Real probability:", round(prob_real * 100, 2), "%")
    print("Reason:", explain_image(prob_fake))

    return {
        "modality": "Image",
        "prediction": label,
        "fake_prob": round(prob_fake * 100, 2),
        "real_prob": round(prob_real * 100, 2),
        "reason": explain_image(prob_fake)
    }

def predict_video_file(video_path):
    video_path = clean_path(video_path)

    if not os.path.exists(video_path):
        print("\n--- VIDEO RESULT ---")
        print("Input is invalid or path not found.")
        return None

    if not is_video_file(video_path):
        print("\n--- VIDEO RESULT ---")
        print("Input is invalid or corrupted.")
        return None

    frames_folder = video_to_frames(video_path)
    if frames_folder is None:
        print("\n--- VIDEO RESULT ---")
        print("Input video is invalid or corrupted.")
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
            prob_fake = class_to_prob.get(1, 0.0)
            probs.append(prob_fake)

    if not probs:
        print("\n--- VIDEO RESULT ---")
        print("Input video is invalid or corrupted.")
        return None

    avg_prob = float(np.mean(probs))
    label = decide_label(avg_prob)

    print("\n--- VIDEO RESULT ---")
    print("Prediction:", label)
    print("Fake probability:", round(avg_prob * 100, 2), "%")
    print("Real probability:", round((1 - avg_prob) * 100, 2), "%")
    print("Reason:", explain_video(avg_prob))

    return {
        "modality": "Video",
        "prediction": label,
        "fake_prob": round(avg_prob * 100, 2),
        "real_prob": round((1 - avg_prob) * 100, 2),
        "reason": explain_video(avg_prob)
    }

def predict_audio_file(audio_path):
    audio_path = clean_path(audio_path)

    if not os.path.exists(audio_path):
        print("\n--- AUDIO RESULT ---")
        print("Input is invalid or path not found.")
        return None

    if not is_audio_file(audio_path):
        print("\n--- AUDIO RESULT ---")
        print("Input is invalid or corrupted.")
        return None

    feat = extract_mfcc_features(audio_path)
    if feat is None:
        print("\n--- AUDIO RESULT ---")
        print("Input audio is invalid or corrupted.")
        return None

    feat = aud_scaler.transform(feat.reshape(1, -1))

    probs = aud_model.predict_proba(feat)[0]
    classes = aud_model.classes_
    class_to_prob = {cls: prob for cls, prob in zip(classes, probs)}

    prob_real = class_to_prob.get(0, 0.0)
    prob_fake = class_to_prob.get(1, 0.0)
    label = decide_label(prob_fake)

    print("\n--- AUDIO RESULT ---")
    print("Prediction:", label)
    print("Fake probability:", round(prob_fake * 100, 2), "%")
    print("Real probability:", round(prob_real * 100, 2), "%")
    print("Reason:", explain_audio(prob_fake))

    return {
        "modality": "Audio",
        "prediction": label,
        "fake_prob": round(prob_fake * 100, 2),
        "real_prob": round(prob_real * 100, 2),
        "reason": explain_audio(prob_fake)
    }

# =========================================================
# MAIN
# =========================================================
print("Give file paths. Press Enter to skip any modality.")

image_path = input("Enter image path: ").strip()
video_path = input("Enter video path: ").strip()
audio_path = input("Enter audio path: ").strip()

results = []

if image_path:
    r = predict_image_file(image_path)
    if r is not None:
        results.append(r)

if video_path:
    r = predict_video_file(video_path)
    if r is not None:
        results.append(r)

if audio_path:
    r = predict_audio_file(audio_path)
    if r is not None:
        results.append(r)

if not results:
    print("\nNo valid input given.")
else:
    print("\n================ RESULT TABLE ================")
    print(f"{'Modality':<10} {'Prediction':<12} {'Fake %':<10} {'Real %':<10}")
    print("-" * 50)

    for r in results:
        print(f"{r['modality']:<10} {r['prediction']:<12} {r['fake_prob']:<10} {r['real_prob']:<10}")

    print("\n================ EXPLANATIONS ================")
    for r in results:
        print(f"{r['modality']}: {r['reason']}")
        # FINAL CROSS-MODAL RESULT
probs_for_fusion = []

for r in results:
    probs_for_fusion.append(r["fake_prob"] / 100.0)

if probs_for_fusion:
    final_score = float(np.mean(probs_for_fusion))
    final_label = decide_label(final_score)

    print("\n================ FINAL CROSS-MODAL RESULT ================")
    print("Final Prediction:", final_label)
    print("Final Fake Probability:", round(final_score * 100, 2), "%")
    print("Final Real Probability:", round((1 - final_score) * 100, 2), "%")