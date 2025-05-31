import os

# Thư mục chứa các tệp cần đổi tên
folder_path = "output_configs_issue"
prefix = "issue_"

# Lặp qua tất cả các tệp trong thư mục
for filename in os.listdir(folder_path):
    old_path = os.path.join(folder_path, filename)
    
    # Kiểm tra xem đó có phải là tệp không
    if os.path.isfile(old_path):
        new_filename = prefix + filename
        new_path = os.path.join(folder_path, new_filename)
        
        # Đổi tên tệp
        os.rename(old_path, new_path)
        print(f"Đã đổi tên: {filename} -> {new_filename}")

print("Hoàn thành!")