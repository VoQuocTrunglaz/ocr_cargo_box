import pytest
from fastapi.testclient import TestClient
from main import app
import os

@pytest.fixture(scope="module")
def test_client():
    yield TestClient(app)  # Trả về một instance TestClient

def test_root(test_client):  # Nhận test_client từ fixture
    response = test_client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello, FastAPI is running!"}

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
        response = test_client.post("/upload/", files=files)

    # Kiểm tra kết quả
    assert response.status_code == 200
    json_response = response.json()
    assert "container_number" in json_response, "Thiếu khóa container_number trong response"
    assert "container_info" in json_response, "Thiếu khóa container_info trong response"