import cv2
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
VIDEO = ROOT / "videos/out2.mp4"

cap = cv2.VideoCapture(str(VIDEO))
cv2.namedWindow("tuner")
cv2.createTrackbar("H low",  "tuner",  0, 180, lambda x: None)
cv2.createTrackbar("H high", "tuner", 30, 180, lambda x: None)
cv2.createTrackbar("S low",  "tuner", 80, 255, lambda x: None)
cv2.createTrackbar("V low",  "tuner", 50, 255, lambda x: None)

paused = False
frame  = None

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

    if frame is None:
        continue

    h_low  = cv2.getTrackbarPos("H low",  "tuner")
    h_high = cv2.getTrackbarPos("H high", "tuner")
    s_low  = cv2.getTrackbarPos("S low",  "tuner")
    v_low  = cv2.getTrackbarPos("V low",  "tuner")

    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (h_low, s_low, v_low), (h_high, 255, 255))

    display = cv2.resize(frame, (640, 360))
    cv2.putText(display, f"HSV low: (q{h_low}, {s_low}, {v_low})  high: ({h_high}, 255, 255)",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("tuner", np.hstack([
        display,
        cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), (640, 360))
    ]))

    key = cv2.waitKey(30) & 0xFF
    if key == ord("q"):
        break
    elif key == ord(" "):
        paused = not paused

cap.release()
cv2.destroyAllWindows()
