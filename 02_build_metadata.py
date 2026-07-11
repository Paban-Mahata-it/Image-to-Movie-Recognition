import os
import cv2
import csv

FRAMES_DIR = "C:/Users/HP/Desktop/VS_Movie_Recognition/Data/Frames"
OUTPUT_CSV = "C:/Users/HP/Desktop/VS_Movie_Recognition/Data/Metadata.csv"

def get_frame_timestamp(frame_index, fps, interval):
    total_seconds = frame_index * interval
    h = int(total_seconds // 3600)
    m = int((total_seconds % 3600) // 60)
    s = int(total_seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def main():
    rows = []

    for currentMovieName in os.listdir(FRAMES_DIR):
        currentMovieFolder = os.path.join(FRAMES_DIR,currentMovieName)

        if not os.path.isdir(currentMovieFolder):
            continue

        # sort the frames by their names
        frame_files = sorted(os.listdir(currentMovieFolder))

        for idx, frame_file in enumerate(frame_files):
            frame_path = os.path.join(currentMovieFolder, frame_file)
            timestamp = get_frame_timestamp(idx, fps=0, interval=1)

            rows.append([currentMovieName,frame_path,timestamp])
    
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["movie_name","frame_path","timestamp"])
        writer.writerows(rows)

    print(f"Metadata Written to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()