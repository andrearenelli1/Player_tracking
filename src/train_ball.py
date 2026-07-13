from ultralytics import YOLO
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA = ROOT / "dataset_ball/data.yaml"


def main() -> None:
    model = YOLO("yolov8s.pt")
    model.train(
        data=str(DATA),
        epochs=50,
        imgsz=(720, 1280),
        batch=8,
        device="cuda",
        patience=20,
    )


if __name__ == "__main__":
    main()
