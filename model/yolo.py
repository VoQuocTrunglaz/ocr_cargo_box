import mlflow
from ultralytics import YOLO
import os
import torch

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("YOLOv11n_Training")

runs = mlflow.search_runs(order_by=["start_time DESC"])
run_number = len(runs) + 1
run_name = f"YOLO_Run_{run_number}"

with mlflow.start_run(run_name=run_name) as run:
    run_id = run.info.run_id  

    model = YOLO("yolo11n.pt")

    results = model.train(data="mydata.yaml", epochs=40)

    mlflow.log_param("epochs", 40)
    mlflow.log_param("data", "mydata.yaml")

    weights_dir = "runs/detect/train/weights"
    best_model_path = os.path.join(weights_dir, "best.pt")

    if os.path.exists(best_model_path):
        best_model = YOLO(best_model_path)

        torchscript_path = best_model.export(format="torchscript")

        if torchscript_path and os.path.exists(torchscript_path):
            print(f"Logging model to MLflow: {torchscript_path}")

            loaded_model = torch.jit.load(torchscript_path)

            model_artifact_path = "yolo_model"
            mlflow.pytorch.log_model(
                pytorch_model=loaded_model,
                artifact_path=model_artifact_path
            )

            mlflow.log_artifact(best_model_path, artifact_path="weights")

            print("✅ Model successfully logged to MLflow!")

            model_uri = f"runs:/{run_id}/{model_artifact_path}"
            result = mlflow.register_model(
                model_uri=model_uri,
                name="YOLOv11n_Training"
            )

            print(f"✅ Model registered successfully! Version: {result.version}")
        else:
            print("⚠️ Model export failed: Could not generate TorchScript model!")
    else:
        print(f"⚠️ Best model not found at: {best_model_path}")
        print("⚠️ Training might have failed or completed without saving weights!")

mlflow.end_run()
