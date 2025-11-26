"""
Process the CVAT dataset by splitting it into training and validation sets
and moving the files to the processed dataset directory.
"""
from remove_frames import remove_frames_by_range
from typing import Tuple, List
from pathlib import Path
import shutil
import random
import yaml


def move_image(image_path: Path, dest_dir: Path) -> Path:
    """
    Move an image file to the destination directory.
    
    Args:
        image_path: Source path to the image file
        dest_dir: Destination directory path
        
    Returns:
        Path to the moved file
    """
    dest_path = dest_dir / image_path.name
    shutil.move(str(image_path), str(dest_path))
    return dest_path


def move_label(image_path: Path, source_labels_dir: Path, dest_labels_dir: Path) -> None:
    """
    Move corresponding label file to destination directory.
    Creates empty label file if source label doesn't exist.
    
    Args:
        image_path: Path to the image file
        source_labels_dir: Source directory containing label files
        dest_labels_dir: Destination directory for label files
    """
    label_filename = image_path.stem + '.txt'
    source_label_path = source_labels_dir / label_filename
    dest_label_path = dest_labels_dir / label_filename
    
    if source_label_path.exists():
        shutil.move(str(source_label_path), str(dest_label_path))
    else:
        # Create empty file to ensure 1:1 mapping between images and labels
        dest_label_path.touch()


def get_image_files(image_dir: Path) -> List[Path]:
    """
    Get all image files (jpg and png) from a directory.
    
    Args:
        image_dir: Directory to search for images
        
    Returns:
        List of image file paths
    """
    jpg_files = list(image_dir.glob("*.jpg"))
    png_files = list(image_dir.glob("*.png"))
    return jpg_files + png_files


def split_files(files: List[Path], train_ratio: float, seed: int = 42) -> Tuple[List[Path], List[Path]]:
    """
    Split a list of files into train and validation sets.
    
    Args:
        files: List of file paths to split
        train_ratio: Ratio of files to use for training (0.0 to 1.0)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_files, val_files)
    """
    files_copy = files.copy() # Shallow copy to avoid modifying the original list
    random.seed(seed)
    random.shuffle(files_copy)
    
    num_train = int(len(files_copy) * train_ratio)
    train_files = files_copy[:num_train]
    val_files = files_copy[num_train:]
    
    return train_files, val_files


def process_class(
    class_name: str,
    cvat_images_dir: Path,
    cvat_labels_dir: Path,
    processed_base_dir: Path,
    train_ratio: float
) -> Tuple[int, int]:
    """
    Process a single class: split images into train/val and move files.
    
    Args:
        class_name: Name of the class directory
        cvat_images_dir: Source directory for class images
        cvat_labels_dir: Source directory for class labels
        processed_base_dir: Base directory for processed dataset
        train_ratio: Ratio of files for training
        
    Returns:
        Tuple of (num_train, num_val) files processed
    """
    # Source paths for this class
    class_images_dir = cvat_images_dir / class_name
    class_labels_dir = cvat_labels_dir / class_name
    
    # Get all image files
    image_files = get_image_files(class_images_dir)
    
    if not image_files:
        print(f"{class_name:20s}: No images found, skipping")
        return 0, 0
    
    # Split into train and validation
    train_files, val_files = split_files(image_files, train_ratio)
    
    # Create destination directories
    train_image_dir = processed_base_dir / "images" / "train" / class_name
    train_label_dir = processed_base_dir / "labels" / "train" / class_name
    val_image_dir = processed_base_dir / "images" / "val" / class_name
    val_label_dir = processed_base_dir / "labels" / "val" / class_name
    
    for dir_path in [train_image_dir, train_label_dir, val_image_dir, val_label_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Process training files
    train_txt_path = processed_base_dir / "train.txt"
    with train_txt_path.open("a") as f:
        for image_path in train_files:
            dest_path = move_image(image_path, train_image_dir)
            # Write relative path with ./ prefix for Colab compatibility
            rel_path = dest_path.relative_to(processed_base_dir)
            f.write(f"./{rel_path}\n")
            move_label(image_path, class_labels_dir, train_label_dir)
    
    # Process validation files
    val_txt_path = processed_base_dir / "val.txt"
    with val_txt_path.open("a") as f:
        for image_path in val_files:
            dest_path = move_image(image_path, val_image_dir)
            # Write relative path with ./ prefix for Colab compatibility
            rel_path = dest_path.relative_to(processed_base_dir)
            f.write(f"./{rel_path}\n")
            move_label(image_path, class_labels_dir, val_label_dir)
    
    print(f"{class_name:20s}: {len(train_files):3d} train, {len(val_files):3d} val "
          f"(total: {len(image_files):3d})")
    
    return len(train_files), len(val_files)


def generate_data_yaml(
    output_dir: Path,
    train_file: str = "train.txt",
    val_file: str = "val.txt"
) -> None:
    """
    Generate a data.yaml file for YOLO training.
    
    Args:
        output_dir: Directory where data.yaml will be created
        train_file: Name of the train file (default: "train.txt")
        val_file: Name of the validation file (default: "val.txt")
    """
    
    # Create names dictionary with the single class "Navel"
    names_dict = {
        0: "Navel"
    }
    
    # Build YAML structure
    data = {
        "train": train_file,
        "val": val_file,
        "names": names_dict,
        "path": "."  # Relative path
    }
    
    # Write YAML file
    yaml_path = output_dir / "data.yaml"
    with yaml_path.open("w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    print(f"Generated data.yaml file at {yaml_path}")


def post_processing_dataset_summary(processed_base_dir: Path) -> None:
    """Print a summary of the dataset after processing."""
    train_dir = processed_base_dir / "images" / "train"
    val_dir = processed_base_dir / "images" / "val"
    
    train_count = sum(1 for _ in train_dir.rglob("*.jpg")) + sum(1 for _ in train_dir.rglob("*.png"))
    val_count = sum(1 for _ in val_dir.rglob("*.jpg")) + sum(1 for _ in val_dir.rglob("*.png"))
    
    print(f"\nPost-processing dataset summary:")
    print(f"  Train images after processing: {train_count}")
    print(f"  Val images after processing: {val_count}")
    print(f"  Total after processing: {train_count + val_count}")

    # Check train.txt, val.txt, and data.yaml exist
    train_txt_path = processed_base_dir / "train.txt"
    val_txt_path = processed_base_dir / "val.txt"
    yaml_path = processed_base_dir / "data.yaml"
    print(f"  train.txt exists: {train_txt_path.exists()}")
    print(f"  val.txt exists: {val_txt_path.exists()}")    
    print(f"  data.yaml exists: {yaml_path.exists()}")


def pre_processing_dataset_summary(processed_base_dir: Path) -> None:
    """Print a summary of the dataset before processing."""
    print(f"Pre-processing dataset summary:")
    total_train = sum(1 for _ in processed_base_dir.rglob("*.jpg")) + sum(1 for _ in processed_base_dir.rglob("*.png"))
    print(f"  Total images: {total_train}\n")


def main():
    """Main function to perform train/validation split."""
    # Configuration
    CVAT_BASE_DIR = Path("/Users/sebnico/Desktop/CIS4900/injection-tracker/cvat_second_dataset")
    
    remove_frames_by_range(CVAT_BASE_DIR / "images" / "Train" / "flashlight", 151, 315)
    pre_processing_dataset_summary(CVAT_BASE_DIR)

    PROCESSED_BASE_DIR = Path("/Users/sebnico/Desktop/CIS4900/injection-tracker/processed_second_dataset")
    TRAIN_RATIO = 0.8
    RANDOM_SEED = 42
    
    # Source directories
    cvat_images_train_dir = CVAT_BASE_DIR / "images" / "Train"
    cvat_labels_train_dir = CVAT_BASE_DIR / "labels" / "Train"
    
    # Create processed base directory
    PROCESSED_BASE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get all class directories
    class_dirs = [d.name for d in cvat_images_train_dir.iterdir() 
                  if d.is_dir()]
    class_dirs.sort()
    
    if not class_dirs:
        print("No class directories found!")
        return
    
    # Calculate split percentage for display
    train_pct = int(TRAIN_RATIO * 100)
    val_pct = 100 - train_pct
    
    print(f"Performing stratified {train_pct}/{val_pct} train/val split...\n")
    print("-" * 60)
    
    total_train = 0
    total_val = 0
    
    # Process each class
    for class_name in class_dirs:
        num_train, num_val = process_class(
            class_name,
            cvat_images_train_dir,
            cvat_labels_train_dir,
            PROCESSED_BASE_DIR,
            TRAIN_RATIO
        )
        total_train += num_train
        total_val += num_val
    
    # Print summary
    print("-" * 60)
    print(f"{'TOTAL':20s}: {total_train:3d} train, {total_val:3d} val "
          f"(total: {total_train + total_val:3d})")
    print(f"\nSplit complete! Files moved to {PROCESSED_BASE_DIR} and split into training and validation sets.")

    # Generate data.yaml file
    generate_data_yaml(PROCESSED_BASE_DIR)

    # Quick validate the dataset
    post_processing_dataset_summary(PROCESSED_BASE_DIR)


if __name__ == "__main__":
    main()