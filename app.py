import os
import cv2
import re
import numpy as np
from fastapi import FastAPI, File, UploadFile
from paddleocr import PaddleOCR
from ultralytics import YOLO
import uvicorn
from rapidfuzz import process, fuzz

# Lấy thư mục chứa file script, sau đó quay về thư mục gốc
BASE_DIR = os.getcwd()
MODEL_DIR = os.path.join(BASE_DIR, "model", "runs", "detect")
UPLOAD_DIR = os.path.join(BASE_DIR, "upload")

def get_latest_model():
    train_folders = sorted(
        [f for f in os.listdir(MODEL_DIR) if f.startswith("train")],
        key=lambda f: os.path.getmtime(os.path.join(MODEL_DIR, f)),
        reverse=True
    )
    if not train_folders:
        raise FileNotFoundError("Không tìm thấy mô hình YOLO!")

    model_path = os.path.join(MODEL_DIR, train_folders[0], "weights", "best.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy model tại: {model_path}")

    return model_path

MODEL_PATH = get_latest_model()

os.makedirs(UPLOAD_DIR, exist_ok=True)
# 🔹 Load mô hình YOLO & PaddleOCR
yolo_model = YOLO(MODEL_PATH)
ocr = PaddleOCR(lang="en",
                det_db_box_thresh=0.3, 
                rec_algorithm="CRNN", 
                use_angle_cls=True,
                det_db_unclip_ratio=1.8)

# 🔹 Khởi tạo ứng dụng FastAPI
app = FastAPI(title="OCR Container API", version="1.0", description="API nhận diện chữ trên container")

def detect_text(image):
    """Phát hiện vùng chữ bằng YOLO"""
    results = yolo_model(image)[0]
    if not results.boxes or len(results.boxes.xyxy) == 0:
        return image, []

    cropped_images = []
    for i, bbox in enumerate(results.boxes.xyxy):
        xmin, ymin, xmax, ymax = map(int, bbox.tolist())
        cropped = image[ymin:ymax, xmin:xmax]
        cropped_images.append((cropped, (xmin, ymin, xmax, ymax)))

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

def extract_container_number(detected_texts):
    """Trích xuất số hiệu container từ danh sách detected_texts"""
    potential_prefix = None
    potential_serial = None
    potential_type = None
    
    container_regex = re.compile(r"^([A-Z]{4})(\d{6})$")  # Tách Prefix + Serial bị dính liền
    prefix_regex = re.compile(r"^[A-Z]{4}$")  # Prefix riêng
    serial_regex = re.compile(r"^\d{6}$")  # Serial riêng
    type_code_regex = re.compile(r"^[A-Z0-9]{3,4}$")  # Type Code hợp lệ
    
    invalid_labels = {"MAX GROSS", "TARE", "NET", "CUBE"}
    
    for i, (text, confidence) in enumerate(detected_texts):
        text = text.strip().upper()

        if text in invalid_labels:
            continue

        # Kiểm tra Prefix + Serial bị dính liền (VD: "TTNU872638")
        match = container_regex.match(text)
        if match:
            potential_prefix, potential_serial = match.groups()
            continue

        # Tìm Prefix riêng (4 chữ cái)
        if prefix_regex.match(text):
            potential_prefix = text
            if i + 1 < len(detected_texts) and serial_regex.match(detected_texts[i + 1][0].strip().upper()):
                potential_serial = detected_texts[i + 1][0].strip().upper()
            continue

        # Tìm Serial Number riêng (6 số)
        if serial_regex.match(text):
            potential_serial = text
            if i > 0 and prefix_regex.match(detected_texts[i - 1][0].strip().upper()):
                potential_prefix = detected_texts[i - 1][0].strip().upper()
            continue

        # Xác định Type Code (3-4 ký tự hợp lệ)
        if type_code_regex.match(text):
            potential_type = text

        if potential_prefix and potential_serial:
            return {
                "prefix": potential_prefix,
                "serial_number": potential_serial,
                "type_code": potential_type
            }
    
    return None

def extract_numbers_from_text(detected_texts, index, container_serial=None):
    """Tìm kiếm số liệu liên quan từ danh sách OCR"""
    numbers = []
    
    for i in range(index + 1, min(index + 4, len(detected_texts))):
        text, _ = detected_texts[i]
        text = text.replace("L8", "LB").replace(",", ".").replace("M3", "CU.M").replace("F1.", "FT.")
        
        if container_serial and container_serial in text:
            continue

        found_numbers = re.findall(r"\d+\.\d+|\d+", text)
        extracted = [float(num) for num in found_numbers if num != container_serial]
        numbers.extend(extracted)
    
    return numbers

def classify_fields(detected_texts):
    """Phân loại dữ liệu trọng lượng, thể tích container"""
    fields = {
        "container_number": extract_container_number(detected_texts),
        "container_info": {
            "Max Gross": {"kg": None, "lbs": None},
            "Tare Weight": {"kg": None, "lbs": None},
            "Max Payload": {"kg": None, "lbs": None},
            "Cube Volume": {"cu.m": None, "cu.ft": None}
        }
    }
    
    label_map = {
        "Max Gross": ["GROSSWE", "GROSSWT", "MAX GROSS", "MAX.GROSS", "MAX.WT"],
        "Tare Weight": ["TARE"],
        "Max Payload": ["PAYLOAD", "MAX.CARGO", "NET"],
        "Cube Volume": ["CUBE", "CU.CAP", "CU CAP"]
    }
    
    MIN_CONFIDENCE = 0.8
    container_number = extract_container_number(detected_texts)
    if container_number:
        fields["container_number"].update(container_number)
        print(f"Container number detected: {container_number}")
    
    valid_texts = [(text.strip(), confidence) for text, confidence in detected_texts if confidence >= MIN_CONFIDENCE]
    
    for i, (text, confidence) in enumerate(valid_texts):
        if container_number:
            container_values = {v for v in container_number.values() if v}
            if any(text.startswith(v) or text.endswith(v) or text in container_values for v in container_values):
                continue

        current_label = None
        for label, field_name in label_map.items():
            best_match = process.extractOne(text, field_name, scorer=fuzz.ratio)
            if best_match and best_match[1] > 70:
                current_label = label
                print(f"Label found: '{text}' -> {label} (Match: {best_match[0]})")
                break

        if current_label:
            numbers = extract_numbers_from_text(valid_texts, i, container_serial=container_number.get("serial_number") if container_number else None)
            print(f"Numbers extracted for {current_label}: {numbers}")
            
            if numbers:
                if current_label == "Cube Volume":
                    fields["container_info"][current_label]["cu.m"] = numbers[0]
                    if len(numbers) > 1:
                        fields["container_info"][current_label]["cu.ft"] = numbers[1]
                else:
                    fields["container_info"][current_label]["kg"] = numbers[0]
                    if len(numbers) > 1:
                        fields["container_info"][current_label]["lbs"] = numbers[1]
    
    return fields

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """API nhận ảnh và trả về dữ liệu OCR"""
    contents = await file.read()

    # Lưu ảnh tải lên vào thư mục static/upload
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(contents)

    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        return {"error": "Không thể đọc ảnh"}

    _, cropped_images = detect_text(image)
    if not cropped_images:
        return {"message": "Không phát hiện chữ"}

    detected_texts = []
    for cropped, bbox in cropped_images:
        texts = recognize_text(cropped)
        detected_texts.extend([(text, conf) for text, conf in texts])

    result = classify_fields(detected_texts)
    return result

@app.get("/")  # 🔹 Route cho trang chủ
def read_root():
    return {"message": "Hello, FastAPI is running!"}

@app.get("/uploads/")
def list_uploaded_files():
    files = os.listdir(UPLOAD_DIR)
    return {"uploaded_files": files}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)