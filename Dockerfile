FROM python:3.10

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip

RUN apt-get update && apt-get install -y libgl1-mesa-glx

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefer-binary -r requirements.txt
    
COPY . .

EXPOSE 8000

CMD ["uvicorn", "ocr_api:app", "--host", "0.0.0.0", "--port", "8000"]
