from ultralytics import YOLO
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA = ROOT / "dataset_ball/data.yaml"


def main() -> None:
    model = YOLO("yolov8m.pt")
    model.train(
        data=str(DATA),
        epochs=100,
        imgsz=3840,
        rect=True,
        batch=2,
        device="cuda",
        patience=15,
        mosaic=0.0,        
        scale=(1.0, 1.3),  
    )


if __name__ == "__main__":
    main()
