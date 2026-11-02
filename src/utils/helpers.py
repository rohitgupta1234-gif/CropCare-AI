"""
CropCare AI - Helper Functions
Common utility functions used across the project.
"""

import os
import json
import hashlib
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd


def generate_id(prefix: str = "") -> str:
    """Generate a unique ID with optional prefix."""
    unique_part = uuid.uuid4().hex[:8]
    if prefix:
        return f"{prefix}_{unique_part}"
    return unique_part


def generate_timestamp_id() -> str:
    """Generate ID based on timestamp."""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def compute_file_hash(filepath: Union[str, Path], algorithm: str = "md5") -> str:
    """Compute hash of a file."""
    hash_func = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def format_confidence(confidence: float) -> str:
    """Format confidence as percentage."""
    return f"{confidence * 100:.1f}%"


def format_number(num: Union[int, float], decimals: int = 2) -> str:
    """Format number with thousand separators."""
    if isinstance(num, int):
        return f"{num:,}"
    return f"{num:,.{decimals}f}"


def safe_filename(filename: str, max_length: int = 200) -> str:
    """Sanitize filename for safe filesystem usage."""
    # Remove path components
    filename = os.path.basename(filename)

    # Replace unsafe characters
    unsafe_chars = '<>:"/\\|?*'
    for char in unsafe_chars:
        filename = filename.replace(char, "_")

    # Trim length
    if len(filename) > max_length:
        name, ext = os.path.splitext(filename)
        filename = name[:max_length - len(ext)] + ext

    return filename


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(filepath: Union[str, Path], default: Any = None) -> Any:
    """Load JSON file with default fallback."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(data: Any, filepath: Union[str, Path], indent: int = 2) -> bool:
    """Save data to JSON file."""
    try:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
        return True
    except Exception:
        return False


def load_csv(filepath: Union[str, Path]) -> Optional[pd.DataFrame]:
    """Load CSV file safely."""
    try:
        return pd.read_csv(filepath)
    except Exception:
        return None


def save_csv(df: pd.DataFrame, filepath: Union[str, Path], index: bool = False) -> bool:
    """Save DataFrame to CSV."""
    try:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=index)
        return True
    except Exception:
        return False


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max."""
    return max(min_val, min(max_val, value))


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation."""
    return a + (b - a) * t


def normalize_array(arr: np.ndarray, min_val: float = 0, max_val: float = 1) -> np.ndarray:
    """Normalize array to range."""
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max == arr_min:
        return np.full_like(arr, min_val)
    return (arr - arr_min) / (arr_max - arr_min) * (max_val - min_val) + min_val


def top_k_indices(arr: np.ndarray, k: int) -> np.ndarray:
    """Get indices of top k values."""
    return np.argpartition(arr, -k)[-k:][::-1]


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute softmax."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def one_hot_encode(indices: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert indices to one-hot encoding."""
    return np.eye(num_classes)[indices]


def class_weights_from_counts(class_counts: Dict[str, int]) -> Dict[str, float]:
    """Compute balanced class weights from counts."""
    total = sum(class_counts.values())
    num_classes = len(class_counts)
    weights = {}
    for class_name, count in class_counts.items():
        weights[class_name] = total / (num_classes * count)
    return weights


def split_list(lst: List, chunk_size: int) -> List[List]:
    """Split list into chunks."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def flatten_dict(d: Dict, parent_key: str = "", sep: str = ".") -> Dict:
    """Flatten nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: Dict, sep: str = ".") -> Dict:
    """Unflatten dictionary."""
    result = {}
    for key, value in d.items():
        parts = key.split(sep)
        target = result
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]
        target[parts[-1]] = value
    return result


def retry(func, max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Retry a function with exponential backoff."""
    import time
    last_exception = None
    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as e:
            last_exception = e
            if attempt < max_attempts - 1:
                time.sleep(delay)
                delay *= backoff
    raise last_exception


def get_git_info() -> Dict[str, str]:
    """Get git repository info."""
    try:
        import subprocess
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()[:8]
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        return {"commit": commit, "branch": branch}
    except Exception:
        return {"commit": "unknown", "branch": "unknown"}


def get_system_info() -> Dict[str, Any]:
    """Get system information."""
    import platform
    import sys
    return {
        "platform": platform.platform(),
        "python_version": sys.version,
        "processor": platform.processor(),
        "architecture": platform.architecture(),
    }


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        return self

    def __exit__(self, *args):
        self.end_time = datetime.now()

    @property
    def elapsed(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

    def __str__(self):
        return f"{self.name}: {format_duration(self.elapsed)}"


def print_section(title: str, width: int = 60):
    """Print a formatted section header."""
    print("\n" + "=" * width)
    print(f" {title}")
    print("=" * width)


def print_kv(key: str, value: Any, indent: int = 2):
    """Print key-value pair with formatting."""
    print(f"{' ' * indent}{key}: {value}")


if __name__ == "__main__":
    # Test helpers
    print("Testing helpers...")

    print(f"ID: {generate_id('test')}")
    print(f"Timestamp ID: {generate_timestamp_id()}")
    print(f"Format size: {format_file_size(1536000)}")
    print(f"Format duration: {format_duration(3661)}")
    print(f"Format confidence: {format_confidence(0.9234)}")
    print(f"Safe filename: {safe_filename('test<>file.jpg')}")
    print(f"Clamp: {clamp(150, 0, 100)}")

    with Timer("Test operation") as t:
        import time
        time.sleep(0.1)
    print(t)

    print("Helpers test passed!")