import cv2
import numpy as np
import subprocess
import os


def detect_highlights(video_path):
    cap = cv2.VideoCapture(video_path) #used to capture the frames of a vid.
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")

    prev_blurred = None
    motion_scores = []
    while True:
        ret, frame = cap.read() #ret is a boolean and frame is a numpy 3d array which stores (height, width, channels(colors like BGR, not RGB, each value lies between 0 and 255))

        if not ret: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convert to grayscale

        current_blurred = cv2.GaussianBlur(gray, (9, 9), 0) #ksize is used to determine the window dimension to blur and sigmaX controls the influence of the surrounding pixels.
        if prev_blurred is not None:
            diff = cv2.absdiff(current_blurred, prev_blurred)
            # cv2.imshow("diff", diff) shows the difference between next and current frames
            motion_score = diff.mean()
            motion_scores.append(motion_score)
        prev_blurred = current_blurred

    fps = cap.get(cv2.CAP_PROP_FPS)
    window_size = int(fps * 2)
    windows = sliding_windows(motion_scores,window_size,10)
    highlight_windows = percentile_threshold(windows)
    merged = merge_windows(highlight_windows)
    timestamps = frames_to_timestamps(merged, fps)

    return timestamps

    cap.release()
    cv2.destroyAllWindows()


def percentile_threshold(windows, percentile = 85):
    if not windows:
        return []
    scores = [w[2] for w in windows]
    threshold = np.percentile(scores, percentile)

    highlight_windows = [
        w for w in windows if w[2] >= threshold
    ]
    print(len(highlight_windows), "highlight windows")
    return highlight_windows


def sliding_windows(motion_scores, window_size, step_size):
    windows = []
    i = 0

    while i + window_size <= len(motion_scores):
        window = motion_scores[i : i + window_size]
        window_score = sum(window) / len(window)

        windows.append((i, i + window_size - 1, window_score))
        i += step_size

    return windows

def merge_windows(windows):
    if not windows:
        return []
    merged = []
    current_start, current_end = windows[0][:2] #tuple unpacking. 0 is the index and :2 slices upto the second index only so that only the first two values are taken into consideration during assignment 
   

    for i in windows[1:]:
        next_start, next_end = i[:2]
        if next_start <= current_end:
            current_end = max(current_end, next_end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = next_start, next_end

    merged.append((current_start, current_end))
    return merged

def frames_to_timestamps(merged_windows, fps, padding=2.0): #added padding to ensure it's not absolutely cut to cut
    timestamps = []

    for start_frame, end_frame in merged_windows:
        start_time = max(0, start_frame / fps - padding)
        end_time = end_frame / fps + padding
        timestamps.append((start_time, end_time))

    return timestamps

def cut_clips(video_path, timestamps, output_dir="clips"):
    os.makedirs(output_dir, exist_ok=True)

    for idx, (start, end) in enumerate(timestamps):
        output_path = os.path.join(output_dir, f"clip_{idx+1}.mp4")

        command = [
            "ffmpeg",
            "-ss", str(start),
            "-to", str(end),
            "-i", video_path,
            "-c", "copy",
            output_path,
            "-y"  # overwrite if exists
        ]

        subprocess.run(command, check=True)





if __name__ == "__main__":
    detect_highlights("./videos/video.mp4")
