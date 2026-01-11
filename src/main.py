from video.motion import detect_highlights
from video.motion import cut_clips
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
VIDEO_PATH = BASE_DIR / "video" / "videos" / "video.mp4"


def main():
    print("GameViz pipeline initialized")
    video = str(VIDEO_PATH)
    print("FFmpeg input:", VIDEO_PATH)

    timestamps = detect_highlights(str(VIDEO_PATH))
    print("Timestamps:", timestamps)
    print("Number of clips:", len(timestamps))

    cut_clips(video, timestamps)

if __name__ == "__main__":
    main()
