"""
CropCare AI - Data Preprocessing Script
Preprocesses the PlantVillage dataset for training.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import prepare_dataset
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def main():
    """Run data preprocessing pipeline."""
    print("=" * 60)
    print("CropCare AI - Data Preprocessing")
    print("=" * 60)

    # Verify raw data exists
    if not RAW_DATA_DIR.exists() or not any(RAW_DATA_DIR.iterdir()):
        print(f"❌ Raw data not found at {RAW_DATA_DIR}")
        print("Please run download_dataset.py first")
        return

    # Run preprocessing
    stats = prepare_dataset(
        raw_dir=RAW_DATA_DIR,
        processed_dir=PROCESSED_DATA_DIR,
    )

    print("\n" + "=" * 60)
    print("Preprocessing Complete!")
    print("=" * 60)

    # Print summary
    val_stats = stats["validation"]
    print(f"Total images: {val_stats['total_images']}")
    print(f"Valid images: {val_stats['valid_images']}")
    print(f"Invalid images: {val_stats['invalid_images']}")

    split_stats = stats["splits"]
    for split_name, split_data in split_stats.items():
        print(f"\n{split_name.capitalize()}: {split_data['total']} images")
        for class_name, count in split_data['class_counts'].items():
            if count > 0:
                print(f"  {class_name}: {count}")


if __name__ == "__main__":
    from pathlib import Path
    main()