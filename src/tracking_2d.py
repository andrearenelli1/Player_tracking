from ultralytics import YOLO
from collections import defaultdict
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

display_videos = False
ROOT = Path(__file__).parent.parent
VIDEOS = {
    "out2": ROOT / "videos/out2.mp4",
    "out4": ROOT / "videos/out4.mp4",
    "out13": ROOT / "videos/out13.mp4",
}

CSVS = {
    "csv0": ROOT / "tracking_results/tracking_2d/positions/2d_positions0.csv",
    "csv1": ROOT / "tracking_results/tracking_2d/positions/2d_positions1.csv",
    "csv2": ROOT / "tracking_results/tracking_2d/positions/2d_positions2.csv",
}

def tracking_2d():
    # Import the model
    #model = YOLO('yolo26n.pt')
    model = YOLO('yolov8s.pt')
    i = -1

    for (_, vid), (_, csv_file) in zip(VIDEOS.items(), CSVS.items()):
        i += 1
        cap = cv2.VideoCapture(vid)
        # Initialize the dataframe used to store the data
        df = pd.DataFrame(columns=["frame", "cam_id", "class_id", "object_id", "u", "v", "w", "h"])
        # Frame counter
        frm_cnt = -1
        # Store the track history
        track_history = defaultdict(lambda: [])
        
        # Loop through the video frames
        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()

            if success:
                frm_cnt += 1
                # Run YOLO26 tracking on the frame, persisting tracks between frames
                result = model.track(frame, persist=True, verbose=False)[0]

                # Get the boxes and track IDs
                if result.boxes and result.boxes.is_track:
                    boxes = result.boxes.xywh.cpu().int()
                    track_ids = result.boxes.id.int().cpu().tolist()
                    classes = result.boxes.cls.int().cpu().tolist()

                    # Visualize the result on the frame
                    if display_videos: frame = result.plot()

                    # Skip everithing but people and sport ball
                    for box, track_id, cls in zip(boxes, track_ids, classes):
                        if cls != 0 and cls != 32:
                            continue

                        # Plot the tracks
                        x, y, w, h = box
                        track = track_history[track_id]
                        track.append((float(x), float(y)))
                        if len(track) > 30:
                            track.pop(0)

                        # Draw the tracking lines
                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(frame, [points], isClosed=False, color=(0,230,0), thickness=5)

                        # Write the positions in the df
                        df.loc[len(df)] = [frm_cnt, f"cam_{i}", cls, track_id, x.item(), y.item(), w.item(), h.item()]

                # Display the annotated frame
                resized_frame = cv2.resize(frame, (1280,800))
                if display_videos: cv2.imshow("YOLO26 Tracking", resized_frame)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # Break the loop if the end of the video is reached
                break

        # Release the video capture object and close the display window
        cap.release()
        cv2.destroyAllWindows()
        # Write the df in a csv
        df.to_csv(csv_file, index=False)



def main():
    tracking_2d()

if __name__ == '__main__':
    main()