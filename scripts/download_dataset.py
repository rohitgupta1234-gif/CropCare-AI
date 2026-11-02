"""
CropCare AI - Dataset Download Script
Downloads and prepares the PlantVillage dataset.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import os
import zipfile
import requests
from tqdm import tqdm
import shutil

from src.config import RAW_DATA_DIR, CLASS_NAMES


# PlantVillage dataset URLs
DATASET_URLS = {
    # Updated URLs - GitHub structure changed
    "github_raw": "https://github.com/spMohanty/PlantVillage-Dataset/archive/refs/heads/master.zip",
    "kaggle": "https://www.kaggle.com/datasets/emmarex/plantdisease/download",
    # Alternative mirrors
    "mirror1": "https://storage.googleapis.com/plantvillage-dataset/color.zip",
    # Academic mirrors
    "mirror2": "https://data.mendeley.com/datasets/tywbtsjrjv/1/files/f9190e5c-9a1f-4b5a-9b5b-5b5b5b5b5b5b",
}

# Expected class folders in the dataset
EXPECTED_CLASSES = CLASS_NAMES


def download_file(url: str, dest_path: Path, chunk_size: int = 8192) -> bool:
    """Download file with progress bar."""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(dest_path, 'wb') as f, tqdm(
            desc=dest_path.name,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False


def extract_zip(zip_path: Path, extract_to: Path) -> bool:
    """Extract zip file with progress."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            members = zip_ref.infolist()
            for member in tqdm(members, desc="Extracting", unit="files"):
                zip_ref.extract(member, extract_to)
        return True
    except Exception as e:
        print(f"Extraction failed: {e}")
        return False


def organize_dataset(extract_dir: Path, target_dir: Path):
    """Organize extracted dataset into class folders."""
    target_dir.mkdir(parents=True, exist_ok=True)

    # The PlantVillage dataset has a specific structure
    # After extraction, it's usually: color/<class_name>/<images>
    color_dir = extract_dir / "color"
    if not color_dir.exists():
        # Try to find the color directory
        for d in extract_dir.rglob("color"):
            if d.is_dir():
                color_dir = d
                break

    if not color_dir.exists():
        print(f"Could not find 'color' directory in {extract_dir}")
        return False

    print(f"Organizing dataset from {color_dir} to {target_dir}")

    # Copy each class folder
    for class_name in EXPECTED_CLASSES:
        src_class_dir = color_dir / class_name
        dst_class_dir = target_dir / class_name

        if src_class_dir.exists():
            dst_class_dir.mkdir(parents=True, exist_ok=True)
            # Copy all images
            for img_file in src_class_dir.iterdir():
                if img_file.suffix.lower() in {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}:
                    shutil.copy2(img_file, dst_class_dir / img_file.name)
            print(f"  ✓ {class_name}: {len(list(dst_class_dir.iterdir()))} images")
        else:
            print(f"  ⚠ Class folder not found: {class_name}")

    return True


def verify_dataset(data_dir: Path) -> dict:
    """Verify dataset integrity."""
    stats = {
        "total_images": 0,
        "classes_found": [],
        "class_counts": {},
        "missing_classes": [],
    }

    for class_name in EXPECTED_CLASSES:
        class_dir = data_dir / class_name
        if class_dir.exists():
            images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.png"))
            stats["class_counts"][class_name] = len(images)
            stats["total_images"] += len(images)
            stats["classes_found"].append(class_name)
        else:
            stats["missing_classes"].append(class_name)

    return stats


def download_kaggle_dataset(data_dir: Path, kaggle_json_path: str = None) -> bool:
    """Download dataset using Kaggle API (requires kaggle.json)."""
    try:
        import kaggle
        if kaggle_json_path:
            os.environ['KAGGLE_CONFIG_DIR'] = str(Path(kaggle_json_path).parent)

        print("Downloading from Kaggle...")
        kaggle.api.dataset_download_files(
            'emmarex/plantdisease',
            path=str(data_dir),
            unzip=True
        )
        return True
    except Exception as e:
        print(f"Kaggle download failed: {e}")
        print("Make sure you have kaggle.json in ~/.kaggle/ or provide path")
        return False


def main():
    """Main download function."""
    print("=" * 60)
    print("CropCare AI - PlantVillage Dataset Downloader")
    print("=" * 60)

    # Create raw data directory
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    if RAW_DATA_DIR.exists() and any(RAW_DATA_DIR.iterdir()):
        stats = verify_dataset(RAW_DATA_DIR)
        if stats["total_images"] > 1000:
            print(f"\nDataset already exists with {stats['total_images']} images")
            response = input("Re-download? (y/N): ").strip().lower()
            if response != 'y':
                print("Skipping download.")
                return

    # Try direct download first
    zip_path = RAW_DATA_DIR.parent / "plantvillage_color.zip"

    print("\nAttempting direct download...")
    for name, url in DATASET_URLS.items():
        if name == "kaggle":
            continue
        print(f"Trying {name}...")
        if download_file(url, zip_path):
            print("Download successful!")
            break
    else:
        print("Direct download failed. Trying Kaggle...")
        if download_kaggle_dataset(RAW_DATA_DIR):
            print("Kaggle download successful!")
            return
        else:
            print("\n[X] All download methods failed.")
            print("\nManual download options:")
            print("1. Download from: https://github.com/spMohanty/PlantVillage-Dataset")
            print("2. Download from Kaggle: https://www.kaggle.com/datasets/emmarex/plantdisease")
            print(f"3. Extract to: {RAW_DATA_DIR}")
            print("   Structure should be: data/raw/<class_name>/*.jpg")
            return

    # Extract
    print("\nExtracting dataset...")
    extract_dir = RAW_DATA_DIR.parent / "temp_extract"
    if extract_zip(zip_path, extract_dir):
        print("Extraction successful!")

        # Organize
        print("\nOrganizing dataset...")
        if organize_dataset(extract_dir, RAW_DATA_DIR):
            print("\nDataset organized successfully!")

            # Verify
            stats = verify_dataset(RAW_DATA_DIR)
            print(f"\nVerification:")
            print(f"  Total images: {stats['total_images']}")
            print(f"  Classes found: {len(stats['classes_found'])}")
            if stats['missing_classes']:
                print(f"  Missing classes: {stats['missing_classes']}")

        # Cleanup
        shutil.rmtree(extract_dir, ignore_errors=True)
        zip_path.unlink(missing_ok=True)
    else:
        print("Extraction failed!")


if __name__ == "__main__":
    main()