import os

folder_path = ".\\httpd\\file"  # thay đường dẫn tới folder bạn chứa các file httpd.conf

def parse_httpd_conf_thucong(path):
    tokens = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('//'):
                continue
            parts = line.split(None, 1)
            if len(parts) == 1:
                tokens.append(f"{parts[0]}=")
            else:
                tokens.append(f"{parts[0]}={parts[1]}")
    return tokens

for filename in os.listdir(folder_path):
    if filename.endswith(".conf"):
        file_path = os.path.join(folder_path, filename)
        try:
            tokens = parse_httpd_conf_thucong(file_path)
            print(f"\n📄 File: {filename}")
            print("\n".join(tokens))
        except Exception as e:
            print(f"❌ Lỗi khi xử lý file {filename}: {e}")
