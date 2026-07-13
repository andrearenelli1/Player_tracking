from ultralytics import YOLO
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA = ROOT / "dataset/data.yaml"

def main() -> None:
    model = YOLO("yolov8s.pt")
    model.train(
        data=str(DATA),
        epochs=150,
        imgsz=(720, 1280),
        batch=8,
        device="cuda",
        cls=4.0,
        patience=10
    )

if __name__ == "__main__":
    main()
