from ultralytics import YOLO
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA = ROOT / "dataset/data.yaml"

def main() -> None:
    model = YOLO("yolov8s.pt")
    model.train(
        data=str(DATA),
        epochs=50,
        imgsz=1280,
        batch=4,
        device="cuda",
        cls=2.0,        # upweight classification loss for ball/person imbalance
    )

if __name__ == "__main__":
    main()
