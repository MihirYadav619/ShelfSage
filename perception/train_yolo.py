"""Train the book detector: YOLOv8n-seg, 39 classes (one per book), on perception/data
(perception/make_dataset.py), on the GPU.

Starts from the COCO-pretrained yolov8n-seg weights (downloaded by ultralytics on first use).
Left-right flips are off: a mirrored cover or spine title never occurs in the scene, and the
title is what tells two books apart. The wrist camera's roll is already in the data.

Writes perception/runs/books/ (weights/best.pt, results.csv, plots).

Usage (project root, venv active):
    python perception/train_yolo.py                 # 60 epochs
    python perception/train_yolo.py --epochs 5      # quick check
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    import torch
    from ultralytics import YOLO

    epochs = int(sys.argv[sys.argv.index("--epochs") + 1]) if "--epochs" in sys.argv else 60
    if not torch.cuda.is_available():
        sys.exit("CUDA is not available: install the CUDA build of PyTorch (see README, Setup)")
    print("training on", torch.cuda.get_device_name(0))
    model = YOLO("yolov8n-seg.pt")
    model.train(data=str(ROOT / "perception/data/books.yaml"), epochs=epochs, imgsz=640, batch=16,
                device=0, workers=4, project=str(ROOT / "perception/runs"), name="books", exist_ok=True,
                fliplr=0.0, patience=15, seed=0, plots=True)
    metrics = model.val(data=str(ROOT / "perception/data/books.yaml"), split="val", device=0,
                        project=str(ROOT / "perception/runs"), name="books_val", exist_ok=True)
    print(f"val mask mAP50 {metrics.seg.map50:.3f}, mask mAP50-95 {metrics.seg.map:.3f}, "
          f"box mAP50 {metrics.box.map50:.3f}")


if __name__ == "__main__":
    main()
