import cv2
import os

MOVIE_FOLDER_DIR = "C:/Users/HP/Desktop/VS_Movie_Recognition/Data/Movies"
FRAME_OUTPUT_DIR = "C:/Users/HP/Desktop/VS_Movie_Recognition/Data/Frames"

FRAME_INTERVAL = 1 #SECONDS

def extract_frames_from_movie(movie_path,output_path,interval_sec):
    video = cv2.VideoCapture(movie_path)

    if not video.isOpened():
        print(f"Failed to open {movie_path}")
        return

    fps = video.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps*interval_sec)

    frame_count = 0
    extracted = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_name = f"frame_{extracted:05d}.jpg"
            save_path = os.path.join(output_path, frame_name)
            cv2.imwrite(save_path, frame)
            extracted += 1

        frame_count += 1

    video.release()
    print(f"Extracted {extracted} frames from: {movie_path}")

def main():
    os.makedirs(FRAME_OUTPUT_DIR, exist_ok=True)

    for movie_file in os.listdir(MOVIE_FOLDER_DIR):
        if not movie_file.lower().endswith((".mp4",".mkv",".avi",".mov")):
            continue

        movie_path = os.path.join(MOVIE_FOLDER_DIR,movie_file)
        movie_name = os.path.splitext(movie_file)[0]

        movie_output_dir = os.path.join(FRAME_OUTPUT_DIR, movie_name)
        os.makedirs(movie_output_dir, exist_ok=True)

        print(f"Processing movie {movie_file}")
        extract_frames_from_movie(movie_path, movie_output_dir, FRAME_INTERVAL)

if __name__ == "__main__":
    main()