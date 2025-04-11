FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --prefer-binary -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "ocr_ui.py", "--server.port=8501", "--server.address=0.0.0.0"]
