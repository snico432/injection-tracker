"""
Remove frames from a directory by frame number range.

When exporting datasets from CVAT, all frames are exported including
unlabelled ones. This utility removes unlabelled frames from specific
directories by specifying a frame number range.

Example:
    Remove frames 151-315 from flashlight directory:
    >>> directory = Path("cvat_second_dataset/images/Train/flashlight")
    >>> remove_frames_by_range(directory, 151, 315)
"""
from pathlib import Path
from typing import Tuple


def remove_frames_by_range(
    directory: Path,
    start_frame: int,
    end_frame: int,
    pattern: str = "frame_{:05d}.png"
) -> Tuple[int, int]:
    """
    Remove files in a frame number range.
    
    Args:
        directory: Directory containing the files
        start_frame: Starting frame number
        end_frame: Ending frame number (inclusive)
        pattern: Filename pattern with {} placeholder for frame number
        
    Returns:
        Tuple of (deleted_count, not_found_count)
    """
    deleted_count = 0
    not_found_count = 0
    
    for frame_number in range(start_frame, end_frame + 1):
        filename = pattern.format(frame_number)
        file_path = directory / filename
        
        if file_path.exists():
            try:
                file_path.unlink()
                print(f"Deleted: {filename}")
                deleted_count += 1
            except OSError as e:
                print(f"Error deleting {filename}: {e}")
                not_found_count += 1
        else:
            print(f"Not found: {filename}")
            not_found_count += 1
    
    return deleted_count, not_found_count


if __name__ == "__main__":
    directory = Path("/Users/sebnico/Desktop/CIS4900/injection-tracker/cvat_second_dataset/images/Train/flashlight")
    deleted, not_found = remove_frames_by_range(directory, 151, 315)
    print(f"\nTotal files deleted: {deleted}")
    if not_found > 0:
        print(f"Files not found: {not_found}")