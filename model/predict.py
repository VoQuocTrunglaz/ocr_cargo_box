import os
from ultralytics import YOLO
from PIL import Image

BASE_DIR = os.getcwd()
MODEL_PATH = os.path.join(BASE_DIR, "model", "runs", "detect", "train", "weights", "best.pt")
IMAGE_PATH = "data/test/images/IMG_20211010-18202277_jpg.rf.a22988c7c34fcff54c16d2545e08514b.jpg"
RESULTS_DIR = os.path.join(BASE_DIR, "results")

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
