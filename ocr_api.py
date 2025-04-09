import os
import cv2
import json
import re
import numpy as np
from paddleocr import PaddleOCR
from ultralytics import YOLO
import google.generativeai as genai
from fastapi import FastAPI,  File, UploadFile, HTTPException, UploadFile

# Cấu hình API Gemini
API_KEY = "AIzaSyAlL5ivuNQnSQxc7UwKxsSrgRygFsetqLo"
genai.configure(api_key=API_KEY)

BASE_DIR = os.getcwd()
MODEL_DIR = os.path.join(BASE_DIR, "model", "runs", "detect")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
MODEL_PATH = os.path.join(MODEL_DIR, "train4", "weights", "best.pt")

# def get_latest_model():
#     train_folders = sorted(
#         [f for f in os.listdir(MODEL_DIR) if f.startswith("train")],
#         key=lambda f: os.path.getmtime(os.path.join(MODEL_DIR, f)),
#         reverse=True
#     )
#     if not train_folders:
#         raise FileNotFoundError("Không tìm thấy mô hình YOLO!")

#     model_path = os.path.join(MODEL_DIR, train_folders[0], "weights", "best.pt")
#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"Không tìm thấy model tại: {model_path}")

#     return model_path

# MODEL_PATH = get_latest_model()

# Khởi tạo Flask
app = FastAPI(title="OCR Container API", version="1.0", description="API nhận diện chữ trên container")
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 🔹 Load mô hình YOLO & PaddleOCR
yolo_model = YOLO(MODEL_PATH)
ocr = PaddleOCR(lang="en",
                det_db_box_thresh=0.2, 
                rec_algorithm="CRNN", 
                det_db_unclip_ratio=1.8)

@app.get('/')
def home():
    return {"message": "Hello, OCR API is running!"}

@app.post("/uploads/")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, detail="File phải là định dạng JPG hoặc PNG")
    
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(filepath, "wb") as buffer:
        buffer.write(await file.read())
    
    result = process_ocr_with_gemini(filepath)
    return result

@app.get("/uploads/")
def list_uploaded_files():
    files = os.listdir(UPLOAD_DIR)
    return {"uploaded_files": files}

def process_ocr_with_gemini(image_path):
    """Chuỗi xử lý đầy đủ: phát hiện chữ, cắt ảnh, nhận diện văn bản, phân loại bằng Gemini"""
    image, cropped_images = detect_text(image_path)
    if not cropped_images:
        return {"message": "No text detected"}
    
    detected_texts = []
    for cropped_path, _ in cropped_images:
        texts = recognize_text(cropped_path)
        detected_texts.extend([text for text, _ in texts])
    
    if not detected_texts:
        return {"message": "No readable text found"}
    
    classification_result = classify_with_gemini(detected_texts)
    return {"classification_result": classification_result}

def detect_text(image_path):
    image = cv2.imread(image_path)
    results = yolo_model(image)[0]
    cropped_images = []
    
    for i, bbox in enumerate(results.boxes.xyxy):
        xmin, ymin, xmax, ymax = map(int, bbox.tolist())
        cropped = image[ymin:ymax, xmin:xmax]
        cropped_path = f"uploads/cropped_{i}.jpg"
        cv2.imwrite(cropped_path, cropped)
        cropped_images.append((cropped_path, (xmin, ymin, xmax, ymax)))
    
    return image, cropped_images

def recognize_text(image_path):
    results = ocr.ocr(image_path, cls=True)
    if not results or results[0] is None:
        return []
    
    recognized_texts = []
    for result in results[0]:
        if len(result) < 2:
            continue
        bbox, (text, confidence) = result
        recognized_texts.append((text, confidence))
    
    return recognized_texts

def classify_with_gemini(texts):
    """Phân loại danh sách văn bản vào JSON với định dạng chi tiết."""
    prompt = (
        "Hãy kiểm tra danh sách dữ liệu sau, sửa các đơn vị đo bị sai (nếu có)"
        "Hãy phân loại danh sách sau thành JSON theo format dưới đây. "
        "Chỉ trả về JSON hợp lệ, không có mô tả, không có text dư thừa.\n\n"
        "FORMAT JSON YÊU CẦU:\n"
        "```json\n"
        "{\n"
        '  "container_number": {\n'
        '    "prefix": "",\n'
        '    "serial": "",\n'
        '    "type_code": ""\n'
        "  },\n"
        '  "container_info": {\n'
        '    "max_gross": {"":"kg": "", "lbs": "" },\n'
        '    "tare_weight": { "kg": "", "lbs": "" },\n'
        '    "max_payload": { "kg": "", "lbs": "" },\n'
        '    "cube_volume": { "m3": "", "cuft": "" }\n'
        "  }\n"
        "}\n"
        "```\n\n"
        "CÁCH PHÂN LOẠI:\n"
        "- `container_number` gồm:\n"
        "  - `prefix`: 4 chữ cái đầu của số container.\n"
        "  - `serial`: 6 chữ số cuối của số container.\n"
        "  - `type_code`: Mã loại container.\n"
        "- `container_info` gồm:\n"
        "  - `max_gross`: Giá trị của MAX.GROSS (cả kg và lbs).\n"
        "  - `tare_weight`: Giá trị của TARE (cả kg và lbs).\n"
        "  - `max_payload`: Giá trị của MAX.PAYLOAD hoặc NET (cả kg và lbs).\n"
        "  - `cube_volume`: Giá trị của CUBE (cả m³ và cuft).\n\n"
        f"🔹 Dữ liệu đầu vào: {texts}\n"
        "🔹 Chỉ trả về JSON hợp lệ!"
    )
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if not json_match:
            return None
        
        return json.loads(json_match.group(0))
    except:
        return None

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
