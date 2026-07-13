from ultralytics import YOLO
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA = ROOT / "dataset/data.yaml"

def main() -> None:
    model = YOLO("yolov8s.pt")
    model.train(
        data=str(DATA),
        epochs=50,
        imgsz=(2160, 3840),
        batch=16,
        device="cuda",
        cls=4.0,
        patience=10
    )

if __name__ == "__main__":
    main()
