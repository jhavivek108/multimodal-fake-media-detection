import os

BASE_PATH = "."

image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
audio_exts = (".wav", ".mp3", ".m4a", ".flac", ".ogg")

folders = ["archive (2)", "archive (3)", "archive (4)"]

for folder in folders:
    folder_path = os.path.join(BASE_PATH, folder)

    img_count = 0
    audio_count = 0
    sample_files = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            low = file.lower()
            if low.endswith(image_exts):
                img_count += 1
            elif low.endswith(audio_exts):
                audio_count += 1

            if len(sample_files) < 10:
                sample_files.append(os.path.join(root, file))

    print("=" * 50)
    print("FOLDER:", folder)
    print("Image files:", img_count)
    print("Audio files:", audio_count)
    print("Sample files:")
    for s in sample_files:
        print(" ", s)