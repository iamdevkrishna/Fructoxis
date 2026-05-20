import shutil
import random
from pathlib import Path

# --- CONFIGURATION ---
SOURCE_DIR = Path(r"E:\Projects\Fructoxis\data\Processed Images_Fruits")
DEST_DIR = Path(r"E:\Projects\Fructoxis\data\Fructoxis_Dataset")

TARGET_FRUITS = ["Apple", "Banana", "Guava", "Orange"]
QUALITY_CLASSES = ["Good Quality_Fruits", "Mixed Quality_Fruits", "Bad Quality_Fruits"]

# Capped at 1000 for balance between speed and high accuracy
SAMPLE_SIZE = 1000


def build_clean_dataset():
    print(f"Creating pristine 12-class dataset in: {DEST_DIR}")

    for quality in QUALITY_CLASSES:
        for fruit in TARGET_FRUITS:
            suffix = quality.split()[0]
            source_folder = SOURCE_DIR / quality / f"{fruit}_{suffix}"

            folder_name = f"{fruit}_{suffix}"
            dest_folder = DEST_DIR / folder_name
            dest_folder.mkdir(parents=True, exist_ok=True)

            if not source_folder.exists():
                print(f"  [!] Missing source folder: {source_folder}")
                continue

            all_images = [f for f in source_folder.iterdir() if
                          f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]

            # --- THE SAFETY NET ---
            # Automatically adjusts if the folder has less than 1000 images
            safe_sample_size = min(SAMPLE_SIZE, len(all_images))
            images_to_copy = random.sample(all_images, safe_sample_size)

            print(f"  -> Copying {len(images_to_copy)} images to {folder_name}...")

            for img_path in images_to_copy:
                shutil.copy2(img_path, dest_folder / img_path.name)

    print("\n[SUCCESS] Dataset flattened and ready for training!")


if __name__ == "__main__":
    build_clean_dataset()