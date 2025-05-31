import os

def remove_comments_from_cnf(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    cleaned_lines = []
    for line in lines:
        # Loại bỏ dòng trống hoặc dòng chỉ chứa comment
        stripped = line.strip()
        if stripped.startswith('#') or stripped.startswith(';') or not stripped:
            continue
        # Loại bỏ comment nằm sau nội dung chính
        line = line.split('#')[0].split(';')[0].rstrip()
        if line:
            cleaned_lines.append(line + '\n')

    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(cleaned_lines)

def process_folder(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.cnf'):
                file_path = os.path.join(root, file)
                remove_comments_from_cnf(file_path)
                print(f"Đã xử lý: {file_path}")

# Sử dụng
folder_path = 'mysql-all'  # <-- thay bằng đường dẫn thật
process_folder(folder_path)
