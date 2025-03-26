import os
import cv2
import json
import re
import numpy as np
from paddleocr import PaddleOCR
from ultralytics import YOLO
from rapidfuzz import process, fuzz

# 🔹 Cấu hình đường dẫn
BASE_DIR = os.getcwd()
MODEL_DIR = os.path.join(BASE_DIR, "model", "runs", "detect")

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

OUTPUT_DIR = os.path.join(BASE_DIR, "model", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 🔹 Load mô hình YOLO và PaddleOCR
yolo_model = YOLO(MODEL_PATH)
ocr = PaddleOCR(
    lang="en",  
    det_db_box_thresh=0.4,  # Ngưỡng phát hiện chữ
    rec_algorithm="CRNN",  # Sử dụng thuật toán nhận diện chữ tốt hơn
    use_angle_cls=True,
    det_db_unclip_ratio=1.8 # Điều chỉnh biên chữ, tránh mất ký tự
)

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

def extract_container_number(detected_texts):
    """Trích xuất số hiệu container từ danh sách detected_texts"""
    potential_prefix = None
    potential_serial = None
    potential_type = None

    container_regex = re.compile(r"^([A-Z]{4})(\d{6})$")  # Tách Prefix + Serial bị dính liền
    prefix_regex = re.compile(r"^[A-Z]{4}$")  # Prefix riêng
    serial_regex = re.compile(r"^\d{6}$")  # Serial riêng
    type_code_regex = re.compile(r"^[A-Z0-9]{3,4}$")  # Type Code hợp lệ

    invalid_labels = {"MAX GROSS", "TARE", "NET", "CU.CAP."}

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
            # Kiểm tra dòng kế tiếp có Serial không
            if i + 1 < len(detected_texts) and serial_regex.match(detected_texts[i + 1][0].strip().upper()):
                potential_serial = detected_texts[i + 1][0].strip().upper()
            continue

        # Tìm Serial Number riêng (6 số)
        if serial_regex.match(text):
            potential_serial = text
            # Kiểm tra dòng trước có Prefix không
            if i > 0 and prefix_regex.match(detected_texts[i - 1][0].strip().upper()):
                potential_prefix = detected_texts[i - 1][0].strip().upper()
            continue

        # Xác định Type Code (3-4 ký tự hợp lệ)
        if type_code_regex.match(text):
            potential_type = text

        # Nếu đã tìm đủ Prefix + Serial + Type Code thì trả về kết quả
        if potential_prefix and potential_serial and potential_type:
            return {
                "prefix": potential_prefix,
                "serial_number": potential_serial,
                "type_code": potential_type
            }

    return None

def extract_numbers_from_text(detected_texts, index):
    """Tìm kiếm số liệu liên quan từ danh sách OCR"""
    numbers = []
    for i in range(index + 1, min(index + 4, len(detected_texts))):
        text, _ = detected_texts[i]
        text = text.replace("L8", "LB")
        text = text.replace("M3", "CU.M")
        text = text.replace(",", ".")
        found_numbers = re.findall(r"\d+\.\d+|\d+", text)
        numbers.extend([float(num) for num in found_numbers])  # Chuyển sang float
    return numbers

def classify_fields(detected_texts):
    """Phân loại thông tin container và gán các giá trị từ OCR"""
    fields = {
        "container_number": {"prefix": None, "serial_number": None, "type_code": None},
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

    # Ưu tiên phát hiện số hiệu container trước
    container_number = extract_container_number(detected_texts)
    if container_number:
        fields["container_number"].update(container_number)
        print(f"Container number detected: {container_number}")

    valid_texts = []  # Danh sách các văn bản hợp lệ
    for text, confidence in detected_texts:
        text = text.upper().strip()
        if confidence < MIN_CONFIDENCE:
            continue  # Bỏ qua văn bản có độ chính xác thấp

        # Kiểm tra nếu là những văn bản không hợp lệ
        if re.match(r"^[A-Z]{3,4}$", text) and confidence < 0.6:
            continue  # Loại bỏ những từ nếu độ chính xác thấp

        valid_texts.append((text, confidence))

    # Tiếp tục xử lý thông tin trọng lượng và thể tích
    for i, (text, confidence) in enumerate(valid_texts):
        text = text.upper().strip()
        # Bỏ qua nếu là số hiệu container
        if container_number and text in container_number.values():
            continue

        current_label = None
        for label, field_name in label_map.items():
            best_match = process.extractOne(text, field_name, scorer=fuzz.ratio)

            if best_match and best_match[1] > 70:
                current_label = label
                print(f"Label found: '{text}' -> {label} (Match: {best_match[0]})")
                break

        if current_label:
            numbers = extract_numbers_from_text(valid_texts, i)
            print(f"Numbers extracted for {current_label}: {numbers}")

            # Kiểm tra xem có một hoặc hai số không
            if len(numbers) > 0:
                if current_label == "Cube Volume":
                    fields["container_info"][current_label]["cu.m"] = numbers[0]
                    if len(numbers) > 1:
                        fields["container_info"][current_label]["cu.ft"] = numbers[1]
                else:
                    fields["container_info"][current_label]["kg"] = numbers[0]
                    if len(numbers) > 1:
                        fields["container_info"][current_label]["lbs"] = numbers[1]
            
    return json.dumps(fields, indent=4, ensure_ascii=False)

def process_ocr(image_path):
    """Chuỗi xử lý đầy đủ: phát hiện chữ, cắt ảnh, nhận diện văn bản"""
    image, cropped_images = detect_text(image_path)

    if not cropped_images:
        print("⚠️ Không có chữ để nhận diện!")
        return

    detected_texts = []
    for cropped_img, bbox in cropped_images:
        texts = recognize_text(cropped_img)
        for text, confidence in texts:
            detected_texts.append((text, confidence, bbox))
            print(f"📌 Văn bản: {text} | 🎯 Độ chính xác: {confidence:.2f}")

    # Vẽ kết quả lên ảnh gốc
    #visualize_results(image, detected_texts)

    json_output = classify_fields([(text, confidence) for text, confidence, _ in detected_texts])
    print("📦 JSON Kết quả:\n", json_output)


# 🔹 Chạy thử nghiệm
image_path = os.path.join(BASE_DIR, "data", "test", "images", "IMG_20210906-15132785_jpg.rf.6602e1e07202f7f034b52c969b47ea5e.jpg")
process_ocr(image_path)
