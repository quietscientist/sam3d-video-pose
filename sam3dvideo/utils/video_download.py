#!/usr/bin/env python3
"""
Utilities for downloading and converting videos from URLs.
"""

import os
import subprocess
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve, Request, urlopen
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for urllib downloads."""

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_video(url: str, output_path: str, show_progress: bool = True) -> str:
    """
    Download a video file from a URL.

    Args:
        url: URL to download from
        output_path: Path to save the downloaded file
        show_progress: Show download progress bar

    Returns:
        Path to downloaded file
    """
    import urllib.request

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading from: {url}")
    print(f"Saving to: {output_path}")

    # Set up custom opener with User-Agent to avoid 403 errors
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent', 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')]
    urllib.request.install_opener(opener)

    if show_progress:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
            urlretrieve(url, output_path, reporthook=t.update_to)
    else:
        urlretrieve(url, output_path)

    print(f"✓ Download complete: {output_path}")
    return str(output_path)


def convert_video_to_mp4(input_path: str, output_path: str = None,
                         codec: str = 'libx264', crf: int = 23,
                         remove_source: bool = False, use_moviepy: bool = True) -> str:
    """
    Convert a video file to MP4 format using moviepy (preferred) or ffmpeg.

    Args:
        input_path: Path to input video file
        output_path: Path for output MP4 file (auto-generated if None)
        codec: Video codec to use (default: libx264)
        crf: Constant Rate Factor for quality (default: 23, lower=better)
        remove_source: Remove source file after conversion
        use_moviepy: Use moviepy instead of system ffmpeg (default: True)

    Returns:
        Path to converted MP4 file
    """
    input_path = Path(input_path)

    if output_path is None:
        output_path = input_path.with_suffix('.mp4')
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nConverting video to MP4...")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")

    # Try moviepy first (includes bundled ffmpeg)
    if use_moviepy:
        try:
            from moviepy import VideoFileClip

            print("Using moviepy for conversion...")
            clip = VideoFileClip(str(input_path))
            clip.write_videofile(
                str(output_path),
                codec=codec,
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                logger=None  # Suppress verbose output
            )
            clip.close()

            print(f"✓ Conversion complete: {output_path}")

            # Remove source file if requested
            if remove_source and input_path != output_path:
                print(f"Removing source file: {input_path}")
                input_path.unlink()

            return str(output_path)

        except ImportError:
            print("moviepy not available, trying system ffmpeg...")
            use_moviepy = False
        except Exception as e:
            print(f"moviepy conversion failed: {e}")
            print("Falling back to system ffmpeg...")
            use_moviepy = False

    # Fallback to system ffmpeg
    if not use_moviepy:
        cmd = [
            'ffmpeg',
            '-i', str(input_path),
            '-c:v', codec,
            '-crf', str(crf),
            '-c:a', 'aac',
            '-b:a', '128k',
            '-movflags', '+faststart',
            '-y',
            str(output_path)
        ]

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✓ Conversion complete: {output_path}")

            if remove_source and input_path != output_path:
                print(f"Removing source file: {input_path}")
                input_path.unlink()

            return str(output_path)

        except subprocess.CalledProcessError as e:
            print(f"✗ FFmpeg error: {e.stderr}")
            raise RuntimeError(f"Failed to convert video: {e}")
        except FileNotFoundError:
            raise RuntimeError("ffmpeg not found. Install with: pip install moviepy OR sudo apt install ffmpeg")


def download_and_convert_video(url: str, output_dir: str = "data",
                               filename: str = None,
                               keep_original: bool = False) -> str:
    """
    Download a video from a URL and convert it to MP4 format.

    Args:
        url: URL to download video from
        output_dir: Directory to save the video (default: "data")
        filename: Output filename (auto-generated from URL if None)
        keep_original: Keep original downloaded file after conversion

    Returns:
        Path to final MP4 file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract filename from URL if not provided
    if filename is None:
        parsed_url = urlparse(url)
        filename = Path(parsed_url.path).name
        if not filename:
            filename = "video.mp4"

    # Download to temporary location
    download_path = output_dir / filename
    downloaded_file = download_video(url, download_path)

    # Convert to MP4 if not already MP4
    if not downloaded_file.lower().endswith('.mp4'):
        mp4_path = Path(downloaded_file).with_suffix('.mp4')
        converted_file = convert_video_to_mp4(
            downloaded_file,
            mp4_path,
            remove_source=not keep_original
        )
        return converted_file
    else:
        print(f"✓ File is already MP4: {downloaded_file}")
        return downloaded_file


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python video_download.py <URL> [output_dir] [filename]")
        sys.exit(1)

    url = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "data"
    filename = sys.argv[3] if len(sys.argv) > 3 else None

    result = download_and_convert_video(url, output_dir, filename)
    print(f"\n✓ Video ready: {result}")
