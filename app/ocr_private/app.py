import os
import cv2
import json
import re
import numpy as np
import boto3
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, HTTPException, Query
from dotenv import load_dotenv
from ultralytics import YOLO
import google.generativeai as genai
import concurrent.futures

# Load .env
load_dotenv()
AWS_REGION = os.getenv("AWS_REGION")
S3_BUCKET = os.getenv("S3_BUCKET")
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_PATH = os.path.join(os.getcwd(), "best.pt")

# Cấu hình Gemini & YOLO
genai.configure(api_key=API_KEY)
model_gemini = genai.GenerativeModel("gemini-2.0-flash")
yolo_model = YOLO(MODEL_PATH)

# AWS S3
s3 = boto3.client("s3", region_name=AWS_REGION)

# FastAPI
app = FastAPI(title="OCR Gemini API", version="2.0")

@app.get("/")
def home():
    return {"message": "OCR Gemini API is running on private EC2!"}

@app.get("/uploads/")
def process_image_from_s3(filename: str = Query(..., description="Tên file ảnh đã upload lên S3")):
    try:
        # Đọc ảnh từ S3 (BytesIO)
        image_bytes = s3.get_object(Bucket=S3_BUCKET, Key=filename)['Body'].read()
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Lỗi tải ảnh từ S3: {str(e)}")

    result = process_ocr_with_gemini(pil_image)
    return result

# --- OCR pipeline ---
def process_ocr_with_gemini(pil_image: Image.Image):
    image_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    _, cropped_images = detect_text(image_cv)

    if not cropped_images:
        return {"message": "Không phát hiện chữ!"}

    def worker(pil_crop):
        return recognize_text_with_gemini(pil_crop)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(worker, cropped_images))

    detected_texts = [text for text in results if text and not text.startswith("ERROR")]

    if not detected_texts:
        return {"message": "Không đọc được chữ!"}

    print("🔹 Texts từ Gemini:", detected_texts)
    return {
        "classification_result": classify_with_gemini(detected_texts)
    }

def detect_text(image_cv):
    results = yolo_model(image_cv)[0]
    cropped_images = []

    for bbox in results.boxes.xyxy:
        xmin, ymin, xmax, ymax = map(int, bbox.tolist())
        cv2.rectangle(image_cv, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cropped = image_cv[ymin:ymax, xmin:xmax]
        pil_crop = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        cropped_images.append(pil_crop)

    return image_cv, cropped_images

def recognize_text_with_gemini(pil_crop: Image.Image):
    try:
        prompt = "Hãy đọc văn bản trong ảnh này. Chỉ trả về đoạn văn bản, không thêm mô tả."
        response = model_gemini.generate_content([prompt, pil_crop])
        return response.text.strip()
    except Exception as e:
        print(f"❌ Lỗi Gemini OCR: {e}")
        return None

def classify_with_gemini(texts):
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
        response = model_gemini.generate_content(prompt)
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        return json.loads(json_match.group(0)) if json_match else {"error": "Không parse được JSON"}
    except Exception as e:
        return {"error": str(e)}

# --- Main ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
