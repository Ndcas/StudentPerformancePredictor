Sử dụng repo:

    Chạy file setup.bat để tự tạo môi trường ảo và cài thư viện

    Khởi động môi trường ảo bằng lệnh activate trong .venv/Script

    Tắt môi trường ảo bằng lệnh deactivate

    Chạy file Python bằng python [tên file].py

Cấu trúc:

    File main.py là file gốc để chạy web

    Đặt biến môi trường trong .env

    Train model trong file riêng trong models, nhớ tạo hàm predict để dự đoán, lưu model trong pretrained

    Giao diện web trong templates, các asset tĩnh khác nằm trong static