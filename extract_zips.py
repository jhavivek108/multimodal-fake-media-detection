import os
import zipfile

zip_files = ["archive (2).zip", "archive (3).zip", "archive (4).zip"]

for z in zip_files:
    if os.path.exists(z):
        extract_folder = z.replace(".zip", "")
        os.makedirs(extract_folder, exist_ok=True)

        with zipfile.ZipFile(z, "r") as zip_ref:
            zip_ref.extractall(extract_folder)

        print(f"Extracted: {z} -> {extract_folder}")
    else:
        print(f"Not found: {z}")