import os
from ultralytics import YOLO
from dotenv import load_dotenv

# Load env variables
load_dotenv()

def train():
    # Paths from environment or defaults
    model_path = os.getenv("MODEL_PATH", "weights/last.pt")
    data_path = os.getenv("DATA_CONFIG", "config.yaml")
    project_path = os.getenv("TRAIN_PROJECT", "trains")
    run_name = os.getenv("RUN_NAME", "run1")

    model = YOLO(model_path)

    model.train(
        data=data_path,
        resume=os.getenv("RESUME", "False").lower() == "true",
        epochs=int(os.getenv("EPOCHS", 100)),
        imgsz=int(os.getenv("IMG_SIZE", 416)),
        batch=int(os.getenv("BATCH_SIZE", 2)),
        device=os.getenv("DEVICE", "0"),
        workers=int(os.getenv("WORKERS", 2)),
        save_period=int(os.getenv("SAVE_PERIOD", 15)),
        project=project_path,
        name=run_name,
        verbose=True,
        seed=0,
        deterministic=True
    )

if __name__ == '__main__':
    train()
