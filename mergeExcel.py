import os
import pandas as pd

# Đường dẫn tới thư mục cần quét
folder_path = "configissues\\mysql"

# Danh sách chứa dữ liệu
data = []

# Duyệt qua tất cả các file trong thư mục
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    
    # Kiểm tra nếu là file (không phải folder)
    if os.path.isfile(file_path):
        try:
            # Đọc nội dung file (chế độ text, bỏ qua binary files)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
        except Exception as e:
            content = f"Không thể đọc file: {e}"
        
        # Thêm vào danh sách
        data.append([filename, content])

# Tạo DataFrame từ dữ liệu
df = pd.DataFrame(data, columns=["Tên file", "Nội dung file"])

# Xuất ra file Excel
output_file = "file_list_mysql_issue.xlsx"
df.to_excel(output_file, index=False, engine="openpyxl")

print(f"Đã lưu danh sách file vào {output_file}")