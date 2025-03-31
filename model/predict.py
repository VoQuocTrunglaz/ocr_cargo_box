import os
from ultralytics import YOLO
from PIL import Image
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]  

# Xác định đường dẫn đến model, ảnh test và thư mục kết quả
MODEL_PATH = Path(__file__).resolve().parent / "runs" / "detect" / "train" / "weights" / "best.pt"
IMAGE_PATH = BASE_DIR / "data" / "test" / "images" / "4_jpg.rf.43e07ad086c9d9c9b06005367b48bf41.jpg"
RESULTS_DIR = BASE_DIR / "model" / "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

existing_files = [f for f in os.listdir(RESULTS_DIR) if f.startswith("predictions_") and f.endswith(".jpg")]
next_index = max([int(f.split("_")[1].split(".")[0]) for f in existing_files] + [0]) + 1
OUTPUT_IMAGE_PATH = os.path.join(RESULTS_DIR, f"predictions_{next_index}.jpg")

model = YOLO(MODEL_PATH)

results = model(IMAGE_PATH)

for r in results:
    print(r.boxes)  
    im_array = r.plot()  
    im = Image.fromarray(im_array[..., ::-1])  
    im.show()  
    im.save(OUTPUT_IMAGE_PATH) 

print(f"✅ Ảnh đã được lưu tại: {OUTPUT_IMAGE_PATH}")
