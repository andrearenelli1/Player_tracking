from ultralytics import YOLO
from collections import defaultdict
import cv2
import numpy as np
import pandas as pd

def tracking_2d():
    # Import the model
    #model = YOLO('yolo26n.pt')
    model = YOLO('yolov8s.pt')
    # Save the path to the videos
    video_path1 = '../videos/out2.mp4'
    video_path2 = '../videos/out4.mp4'
    video_path3 = '../videos/out13.mp4'
    video_paths = [video_path1, video_path2, video_path3]

    for i in range(len(video_paths)):
        
        cap = cv2.VideoCapture(video_paths[i])
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
                result = model.track(frame, persist=True)[0]

                # Get the boxes and track IDs
                if result.boxes and result.boxes.is_track:
                    boxes = result.boxes.xywh.cpu()
                    track_ids = result.boxes.id.int().cpu().tolist()
                    classes = result.boxes.cls.int().cpu().tolist()

                    # Visualize the result on the frame
                    frame = result.plot()

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
        df.to_csv(f"../tracking_results/tracking_2d/2d_positions{i}.csv", index=False)



def main():
    tracking_2d()

if __name__ == '__main__':
    main()