import os
import json

# Thư mục chứa các file cần kiểm tra
folder_path = "F:\Study\DACN\dataset\configfiles\hadoop+hbase\configs_q"

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)

    # Bỏ qua nếu không phải là file thường
    if not os.path.isfile(file_path):
        continue

    # Bỏ qua nếu đã có đuôi .json
    if filename.lower().endswith('.json'):
        continue

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            json_obj = json.loads(content)  # thử parse JSON

        # Nếu không có lỗi, thì đây là file JSON hợp lệ
        new_name = filename + ".json" if '.' not in filename else filename.rsplit('.', 1)[0] + ".json"
        new_path = os.path.join(folder_path, new_name)

        os.rename(file_path, new_path)
        print(f"✅ Đã đổi tên {filename} -> {new_name}")

    except (json.JSONDecodeError, UnicodeDecodeError):
        print(f"❌ {filename} không phải là file JSON hợp lệ.")
