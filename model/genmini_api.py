import os
import cv2
import re
import json
import numpy as np
from paddleocr import PaddleOCR
from ultralytics import YOLO
import google.generativeai as genai
from pathlib import Path

# 🔹 Cấu hình API Gemini
API_KEY = "AIzaSyAlL5ivuNQnSQxc7UwKxsSrgRygFsetqLo"
genai.configure(api_key=API_KEY)

BASE_DIR = Path(__file__).resolve().parents[1]  

# Xác định đường dẫn đến model, ảnh test và thư mục kết quả
MODEL_PATH = Path(__file__).resolve().parent / "runs" / "detect" / "train" / "weights" / "best.pt"

# 🔹 Load mô hình YOLO và PaddleOCR
yolo_model = YOLO(MODEL_PATH)  
ocr = PaddleOCR(lang="en", use_angle_cls=True, rec_algorithm="CRNN")

# Thư mục lưu vùng ảnh cắt
OUTPUT_DIR = "cropped_text_regions"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def detect_text(image_path):
    """Phát hiện vùng chứa chữ bằng YOLO"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"❌ Không thể đọc ảnh {image_path}")

    results = yolo_model(image)[0]  # Chạy YOLO
    if results.boxes is None or len(results.boxes.xyxy) == 0:
        print("⚠️ Không phát hiện vùng chứa chữ!")
        return image, []

    cropped_images = []
    for i, bbox in enumerate(results.boxes.xyxy):
        xmin, ymin, xmax, ymax = map(int, bbox.tolist())
        cropped = image[ymin:ymax, xmin:xmax]
        
        # Lưu vùng cắt
        cropped_path = os.path.join(OUTPUT_DIR, f"cropped_{i}.jpg")
        cv2.imwrite(cropped_path, cropped)
        cropped_images.append((cropped_path, (xmin, ymin, xmax, ymax)))

    return image, cropped_images

def recognize_text(image_path):
    """Nhận diện văn bản từ ảnh bằng PaddleOCR với xử lý lỗi"""
    results = ocr.ocr(image_path, cls=True)

    # Kiểm tra kết quả trả về có hợp lệ không
    if not results or results[0] is None:
        print("⚠️ Không tìm thấy văn bản trong ảnh!")
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
        "  - `max_payload`: Giá trị của MAX.PAYLOAD (cả kg và lbs).\n"
        "  - `cube_volume`: Giá trị của CUBE (cả m³ và cuft).\n\n"
        f"🔹 Dữ liệu đầu vào: {texts}\n"
        "🔹 Chỉ trả về JSON hợp lệ!"
    )
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)

        if not response or not response.text:
            print("⚠️ Không nhận được phản hồi từ Gemini")
            return None

        # 🛠️ **Trích xuất JSON từ phản hồi**
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if not json_match:
            print("❌ Lỗi: Gemini không trả về JSON hợp lệ!")
            return None

        json_result = json.loads(json_match.group(0))  # Chuyển đổi thành dict
        return json_result

    except json.JSONDecodeError:
        print("❌ Lỗi khi parse JSON từ Gemini!")
        return None
    except Exception as e:
        print("❌ Lỗi khi gọi API Gemini:", str(e))
        return None

def process_ocr_with_gemini(image_path):
    """Chuỗi xử lý đầy đủ: phát hiện chữ, cắt ảnh, nhận diện văn bản, phân loại bằng Gemini"""
    image, cropped_images = detect_text(image_path)
    if not cropped_images:
        return
    
    detected_texts = []
    for cropped_path, _ in cropped_images:
        texts = recognize_text(cropped_path)
        detected_texts.extend([text for text, _ in texts])
    
    if not detected_texts:
        print("⚠️ Không có văn bản để phân loại!")
        return
    
    # Gọi API Gemini để phân loại
    classification_result = classify_with_gemini(detected_texts)
    print("📦 Kết quả phân loại từ Gemini:\n", classification_result)

# 🔹 Chạy thử nghiệm
image_path = BASE_DIR / "data" / "test" / "images" / "4_jpg.rf.43e07ad086c9d9c9b06005367b48bf41.jpg" # Cập nhật đường dẫn ảnh
process_ocr_with_gemini(image_path)
