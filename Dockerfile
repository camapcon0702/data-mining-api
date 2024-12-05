# Sử dụng Python 3.8 làm base image
FROM python:3.10-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Sao chép requirements.txt vào container
COPY requirements.txt /app/

# Cài đặt các dependencies từ requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ mã nguồn vào container
COPY . /app/

# Mở cổng cho Flask API (Flask mặc định chạy trên cổng 5000)
EXPOSE 8080

# Chạy ứng dụng Flask
CMD ["python", "api.py"]
