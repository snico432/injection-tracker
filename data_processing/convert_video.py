"""
Convert a video file to a directory of image frames.

Extracts frames from a video file using ffmpeg and saves them as
sequentially numbered image files (PNG or JPG format).

Example:
    Extract frames at 3 fps from a video:
    >>> video_path = Path("./flashlight.mp4")
    >>> output_dir = Path("./frames/flashlight")
    >>> extract_frames_ffmpeg(video_path, output_dir, fps=3)
"""

from pathlib import Path
import subprocess
from typing import Optional


def extract_frames_ffmpeg(
    video_path: Path,
    output_folder: Path,
    fps: float = 3.0,
    format: str = "png"
) -> bool:
    """
    Extract frames from a video file using ffmpeg.
    
    Args:
        video_path: Path to the input video file
        output_folder: Directory where frames will be saved
        fps: Frames per second to extract (default: 3.0)
        format: Output image format - 'png' or 'jpg' (default: 'png')
        
    Returns:
        True if successful, False otherwise
    """
    # Validate inputs
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return False
    
    if format not in ['png', 'jpg']:
        print(f"Error: Format must be 'png' or 'jpg', got '{format}'")
        return False
    
    # Create output directory
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Determine output pattern based on format
    output_pattern = output_folder / f"frame_%05d.{format}"
    
    # Build ffmpeg command
    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-y",  # Overwrite output files without asking
        str(output_pattern)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        print(f"Successfully extracted frames to {output_folder}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running ffmpeg: {e}")
        print(f"stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        print("Error: ffmpeg not found. Please install ffmpeg.")
        return False


if __name__ == "__main__":
    name = "flashlight"
    video_path = Path(f"./{name}.mp4")
    output_dir = Path(f"./frames/{name}")
    
    extract_frames_ffmpeg(video_path, output_dir, fps=3.0, format="png")