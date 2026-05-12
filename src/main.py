from ultralytics import YOLO
import cv2

def track():
    model = YOLO('yolo26n.pt')
    results = model.track(
        source='videos/vid_test.mp4', 
        tracker='botsort.yaml', 
        conf=0.1, 
        iou=0.7,
        show=True, 
        stream=True)
    
    for r in results:
        pass


def main():
    track()

if __name__ == '__main__':
    main()