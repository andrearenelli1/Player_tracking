import cv2
from ultralytics import YOLO

# Load the pretrained YOLOv8n model
model = YOLO('yolov8n.pt')

def capture_video_test():

    # Open the default camera
    cam = cv2.VideoCapture(0)

    # Get the default frame width and height
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

    while True:
        ret, frame = cam.read()

        if not ret:
            break

        # Write the frame to the output file
        out.write(frame)

        # Display the captured frame
        cv2.imshow('Camera', frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the capture and writer objects
    cam.release()
    out.release()
    cv2.destroyAllWindows()

def yolo_test():
    # Run inference on the recorded video and save an annotated MP4.
    video = cv2.VideoCapture('output.mp4')
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 20.0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_yolo.mp4', fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = video.read()

        if not ret:
            break

        results = model(frame, conf=0.4, verbose=False)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

    video.release()
    out.release()

def main():
    capture_video_test()
    yolo_test()

if __name__ == '__main__':
    main()