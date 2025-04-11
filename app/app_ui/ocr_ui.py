import streamlit as st
import requests
import boto3
import os
from dotenv import load_dotenv
from io import BytesIO
from PIL import Image
import pandas as pd

# Load environment variables
load_dotenv()
OCR_API_URL = os.getenv("PRIVATE_EC2_API")
S3_BUCKET = os.getenv("S3_BUCKET")
AWS_REGION = os.getenv("AWS_REGION")

# Initialize S3 client
s3 = boto3.client("s3", region_name=AWS_REGION)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 36px;
        font-weight: bold;
        color: #1E90FF;
        text-align: center;
        margin-bottom: 20px;
    }
    .success-message {
        background-color: #E6F4EA;
        color: #2E7D32;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .error-message {
        background-color: #FFEBEE;
        color: #D32F2F;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .warning-message {
        background-color: #FFF3E0;
        color: #F57C00;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #1E90FF;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Main title
st.markdown('<div class="main-title">🚀 Container Image OCR</div>', unsafe_allow_html=True)

# File uploader
with st.container():
    uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "png", "jpeg"], help="Upload a container image to extract text.")

# Upload and Analyze button
if uploaded_file and st.button("Upload and Analyze"):
    # Đọc dữ liệu ảnh vào bộ nhớ để sử dụng nhiều lần
    image_data = uploaded_file.read()  # Đọc dữ liệu ảnh
    filename = uploaded_file.name
    
    # Upload lên S3
    with st.spinner("⏳ Uploading to S3..."):
        s3.upload_fileobj(BytesIO(image_data), S3_BUCKET, filename)
    
    st.markdown(f'<div class="success-message">✅ Image Uploaded: {filename}</div>', unsafe_allow_html=True)

    # Chia giao diện thành 2 cột
    col1, col2 = st.columns([1, 1])

    # Cột trái: hiển thị ảnh gốc
    with col1:
        with st.container():
            st.markdown("### 📷 Uploaded Image")
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(Image.open(BytesIO(image_data)), caption="Original Image", use_container_width=True)
            # Nút tải xuống ảnh
            st.download_button(
                label="📥 Download Image",
                data=image_data,
                file_name=filename,
                mime="image/jpeg"
            )
            st.markdown('</div>', unsafe_allow_html=True)

    # Cột phải: hiển thị kết quả OCR dưới dạng bảng
    with col2:
        with st.container():
            st.markdown("### 📊 OCR Result")
            with st.spinner("⏳ Processing with OCR..."):
                try:
                    # Gọi API OCR
                    res = requests.get(OCR_API_URL, params={"filename": filename})
                    res.raise_for_status()  # Kiểm tra lỗi HTTP
                    data = res.json()

                    # Kiểm tra nếu API trả về message (không có chữ)
                    if "message" in data:
                        st.markdown(f'<div class="warning-message">⚠️ {data["message"]}</div>', unsafe_allow_html=True)
                    else:
                        # Dữ liệu OCR
                        classification_result = data.get("classification_result", {})
                        raw_texts = data.get("raw_texts", [])

                        # Chuẩn bị dữ liệu cho bảng
                        table_data = {
                            "Field": [],
                            "Value": []
                        }

                        # Container Number
                        container_number = classification_result.get("container_number", {})
                        table_data["Field"].extend(["Container Number - Prefix", "Container Number - Serial", "Container Number - Type Code"])
                        table_data["Value"].extend([
                            container_number.get("prefix", ""),
                            container_number.get("serial", ""),
                            container_number.get("type_code", "")
                        ])

                        # Container Info
                        container_info = classification_result.get("container_info", {})
                        max_gross = container_info.get("max_gross", {})
                        tare_weight = container_info.get("tare_weight", {})
                        max_payload = container_info.get("max_payload", {})
                        cube_volume = container_info.get("cube_volume", {})

                        table_data["Field"].extend([
                            "Max Gross (kg)", "Max Gross (lbs)",
                            "Tare Weight (kg)", "Tare Weight (lbs)",
                            "Max Payload (kg)", "Max Payload (lbs)",
                            "Cube Volume (m³)", "Cube Volume (cuft)"
                        ])
                        table_data["Value"].extend([
                            max_gross.get("kg", ""),
                            max_gross.get("lbs", ""),
                            tare_weight.get("kg", ""),
                            tare_weight.get("lbs", ""),
                            max_payload.get("kg", ""),
                            max_payload.get("lbs", ""),
                            cube_volume.get("m3", ""),
                            cube_volume.get("cuft", "")
                        ])

                        # Tạo DataFrame từ table_data
                        df = pd.DataFrame(table_data)

                        # Hiển thị bảng
                        st.markdown('<div class="result-container">', unsafe_allow_html=True)
                        st.table(df)
                        # Nút tải xuống bảng dưới dạng CSV
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Table as CSV",
                            data=csv,
                            file_name="ocr_result.csv",
                            mime="text/csv"
                        )
                        st.markdown('</div>', unsafe_allow_html=True)

                except requests.exceptions.RequestException as e:
                    st.markdown(f'<div class="error-message">❌ API Error: {e}</div>', unsafe_allow_html=True)
                except ValueError as e:
                    st.markdown(f'<div class="error-message">❌ JSON Parsing Error: {e}</div>', unsafe_allow_html=True)