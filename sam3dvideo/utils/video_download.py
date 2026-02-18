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


def _get_ffmpeg_binary() -> str:
    """Find ffmpeg binary: prefer imageio_ffmpeg bundled binary, fall back to system."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return 'ffmpeg'


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


def validate_conversion_brightness(input_path: str, output_path: str,
                                   ffmpeg_bin: str = None,
                                   threshold: float = 0.15) -> bool:
    """
    Check that conversion didn't significantly darken the video by comparing
    mean brightness of a frame from source vs output.

    Returns True if brightness looks OK, False if there's a problem.
    """
    import numpy as np

    if ffmpeg_bin is None:
        ffmpeg_bin = _get_ffmpeg_binary()

    def get_mean_brightness(video_path: str) -> float:
        """Extract a frame at 1s and compute mean pixel intensity."""
        cmd = [
            ffmpeg_bin,
            '-ss', '0.5',
            '-i', str(video_path),
            '-frames:v', '1',
            '-f', 'rawvideo',
            '-pix_fmt', 'rgb24',
            '-v', 'error',
            'pipe:1'
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=30)
            if result.returncode != 0 or len(result.stdout) == 0:
                return -1.0
            frame = np.frombuffer(result.stdout, dtype=np.uint8)
            return float(frame.mean()) / 255.0
        except Exception:
            return -1.0

    src_brightness = get_mean_brightness(input_path)
    dst_brightness = get_mean_brightness(output_path)

    if src_brightness < 0 or dst_brightness < 0:
        print("  ⚠ Could not validate brightness (frame extraction failed)")
        return True  # can't validate, assume OK

    diff = abs(src_brightness - dst_brightness)
    ratio = dst_brightness / max(src_brightness, 0.01)

    print(f"  Brightness check: source={src_brightness:.3f}, output={dst_brightness:.3f}, "
          f"ratio={ratio:.2f}")

    if diff > threshold or ratio < 0.7:
        print(f"  ⚠ WARNING: Significant brightness change detected "
              f"(diff={diff:.3f}, ratio={ratio:.2f})")
        return False

    print(f"  ✓ Brightness OK")
    return True


def convert_video_to_mp4(input_path: str, output_path: str = None,
                         codec: str = 'libx264', crf: int = 18,
                         remove_source: bool = False) -> str:
    """
    Convert a video file to MP4 format using ffmpeg directly.

    Uses ffmpeg with explicit color space handling to avoid color profile
    issues (e.g. limited-range YUV being misinterpreted as full-range).

    Args:
        input_path: Path to input video file
        output_path: Path for output MP4 file (auto-generated if None)
        codec: Video codec to use (default: libx264)
        crf: Constant Rate Factor for quality (default: 18, lower=better)
        remove_source: Remove source file after conversion

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

    ffmpeg_bin = _get_ffmpeg_binary()

    # Use ffmpeg directly with explicit color handling.
    # -pix_fmt yuv420p: standard pixel format for broad compatibility
    # -colorspace bt709 -color_trc bt709 -color_primaries bt709: explicit BT.709
    # -color_range tv: preserve limited range (16-235) to avoid dark output
    cmd = [
        ffmpeg_bin,
        '-i', str(input_path),
        '-c:v', codec,
        '-crf', str(crf),
        '-pix_fmt', 'yuv420p',
        '-colorspace', 'bt709',
        '-color_trc', 'bt709',
        '-color_primaries', 'bt709',
        '-color_range', 'tv',
        '-c:a', 'aac',
        '-b:a', '128k',
        '-movflags', '+faststart',
        '-y',
        str(output_path)
    ]

    print(f"Using ffmpeg: {ffmpeg_bin}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)

        print(f"✓ Conversion complete: {output_path}")

        # Validate brightness wasn't mangled by conversion
        brightness_ok = validate_conversion_brightness(
            str(input_path), str(output_path), ffmpeg_bin
        )
        if not brightness_ok:
            print("  Retrying conversion with explicit color range conversion...")
            # Retry with explicit scale filter to normalize color range
            cmd_retry = [
                ffmpeg_bin,
                '-i', str(input_path),
                '-vf', 'scale=in_range=tv:out_range=pc,scale=in_range=pc:out_range=tv',
                '-c:v', codec,
                '-crf', str(crf),
                '-pix_fmt', 'yuv420p',
                '-colorspace', 'bt709',
                '-color_trc', 'bt709',
                '-color_primaries', 'bt709',
                '-color_range', 'tv',
                '-c:a', 'aac',
                '-b:a', '128k',
                '-movflags', '+faststart',
                '-y',
                str(output_path)
            ]
            result2 = subprocess.run(cmd_retry, capture_output=True, text=True, timeout=300)
            if result2.returncode != 0:
                print(f"  ⚠ Retry also failed, keeping first conversion")
            else:
                validate_conversion_brightness(
                    str(input_path), str(output_path), ffmpeg_bin
                )

        if remove_source and input_path != output_path:
            print(f"Removing source file: {input_path}")
            input_path.unlink()

        return str(output_path)

    except subprocess.CalledProcessError as e:
        stderr = e.stderr if e.stderr else ''
        print(f"✗ FFmpeg error: {stderr[:500]}")
        raise RuntimeError(f"Failed to convert video: {e}")
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found. Install with: pip install imageio-ffmpeg OR sudo apt install ffmpeg"
        )


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
