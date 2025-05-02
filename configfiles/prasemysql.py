import os
from configparser import ConfigParser

def parse_cnf_custom(path):
    tokens = []
    current_section = "default"
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith(';'):
                continue  # Bỏ qua dòng trống hoặc comment
            if line.startswith('[') and line.endswith(']'):
                current_section = line[1:-1].strip()
                continue
            if '=' in line:
                key, value = line.split('=', 1)
                tokens.append(f"{current_section}.{key.strip()}={value.strip()}")
            else:
                tokens.append(f"{current_section}.{line}")
    return tokens

folder_path = ".\\mysql\\file"  # hoặc đường dẫn tới folder chứa các file .cnf

for filename in os.listdir(folder_path):
    if filename.endswith(".cnf"):
        file_path = os.path.join(folder_path, filename)
        try:
            tokens = parse_cnf_custom(file_path)
            print(f"\nFile: {filename}")
            print("\n".join(tokens))
        except Exception as e:
            print(f"❌ Lỗi khi xử lý file {filename}: {e}")