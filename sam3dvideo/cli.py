"""Command-line interface for sam3d-video-pose."""
import sys
from pathlib import Path


def main():
    """Entry point for sam3dvideo CLI command."""
    # Import the process_video script
    scripts_dir = Path(__file__).parent.parent / "scripts"
    sys.path.insert(0, str(scripts_dir))

    try:
        from process_video import main as process_main
        process_main()
    except ImportError as e:
        print(f"Error importing process_video: {e}")
        print("\nPlease ensure the package is installed correctly:")
        print("  pip install -e .")
        sys.exit(1)


if __name__ == "__main__":
    main()
