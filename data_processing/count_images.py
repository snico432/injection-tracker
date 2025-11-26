"""
Count images/frames within subdirectories of a given directory.

Counts all JPG and PNG files in each subdirectory and prints a summary
showing the count per directory and total count.

Args:
    directory: Path to the directory containing subdirectories with images

Example:
    >>> from pathlib import Path
    >>> directory = Path("./cvat_second_dataset/images/Train")
    >>> count_images(directory)
    Image counts per directory:
    
    --------------------------------------------------
    1                   :   76 images
    2                   :  113 images
    3                   :  157 images
    bathroom            :  116 images
    flashlight          :  150 images
    negatives           :   47 images
    outdoors            :   70 images
    --------------------------------------------------
    TOTAL               :  729 images
"""

from pathlib import Path


def count_images(directory: Path) -> None:
    """
    Count images in subdirectories and print summary.
    
    Args:
        directory: Path to directory containing image subdirectories
    """
    # Get all subdirectories
    subdirs = [d.name for d in directory.iterdir() 
               if d.is_dir()]
    # Sort for consistent output
    subdirs.sort()

    print("Image counts per directory:\n")
    print("-" * 50) 

    total_images = 0

    for subdir in subdirs:
        subdir_path = directory / subdir
        jpg_files = list(subdir_path.glob("*.jpg"))
        png_files = list(subdir_path.glob("*.png"))
        count = len(jpg_files) + len(png_files)

        print(f"{subdir:20s}: {count:4d} images")
        total_images += count

    print("-" * 50)
    print(f"{'TOTAL':20s}: {total_images:4d} images")


if __name__ == "__main__":
    directory = Path("/Users/sebnico/Desktop/CIS4900/injection-tracker/cvat_second_dataset/images/Train")
    count_images(directory)