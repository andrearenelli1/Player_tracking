from ultralytics import YOLO
from collections import defaultdict
import cv2
import numpy as np
import pandas as pd

def tracking_2d():
    # Import the model
    model = YOLO('yolo26n.pt')
    # Open the video file
    video_path = 'videos/vid_test.mp4'
    cap = cv2.VideoCapture(video_path)

    # Initialize the dataframe used to store the data
    df = pd.DataFrame(columns=["frame", "cam_id", "object_id", "u", "v"])

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
            result = model.track(frame, persist=True)[0]

            # Get the boxes and track IDs
            if result.boxes and result.boxes.is_track:
                boxes = result.boxes.xywh.cpu()
                track_ids = result.boxes.id.int().cpu().tolist()

                # Visualize the result on the frame
                frame = result.plot()

                # Plot the tracks
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))
                    if len(track) > 30:
                        track.pop(0)

                    # Draw the tracking lines
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [points], isClosed=False, color=(0,230,0), thickness=5)

                    # Write the positions in the df
                    df.loc[len(df)] = [frm_cnt, "cam_1", track_id, x.item(), y.item()]

            # Display the annotated frame
            resized_frame = cv2.resize(frame, (1280,800))
            cv2.imshow("YOLO26 Tracking", resized_frame)

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
    df.to_csv("../tracking_results/tracking_2d/2d_positions.csv", index=False)



def main():
    tracking_2d()

if __name__ == '__main__':
    main()