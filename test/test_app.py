import pytest
from fastapi.testclient import TestClient
from ocr_api import app
import os

@pytest.fixture(scope="module")
def test_client():
    client = TestClient(app)
    yield client  # Chạy test
    client.close() 

def test_root(test_client):  # Nhận test_client từ fixture
    response = test_client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello, OCR API is running!"}

def test_list_uploaded_files(test_client):  # Nhận test_client từ fixture
    response = test_client.get("/uploads/")
    assert response.status_code == 200
    assert "uploaded_files" in response.json()

def test_upload_valid_image(test_client):
    image_path = "data/test/images/4_jpg.rf.43e07ad086c9d9c9b06005367b48bf41.jpg"
    
    # Kiểm tra ảnh test có tồn tại không
    assert os.path.exists(image_path), f"Ảnh test không tồn tại tại đường dẫn: {image_path}"
    
    # Đọc file và gửi request
    with open(image_path, "rb") as image_file:
        files = {"file": ("4_jpg.rf.43e07ad086c9d9c9b06005367b48bf41.jpg", image_file, "image/jpeg")}
        response = test_client.post("/uploads/", files=files)

    # Kiểm tra kết quả
    assert response.status_code == 200
    json_response = response.json()