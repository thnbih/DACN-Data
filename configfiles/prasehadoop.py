import os
import json

def parse_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    tokens = []
    def walk(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, f"{prefix}{k}.")
        else:
            tokens.append(f"{prefix[:-1]}={obj}")
    walk(data)
    return tokens

# Duyệt qua folder "hadoop" và xử lý từng file JSON
folder_path = ".\hadoop+hbase\configs_q+a"

for filename in os.listdir(folder_path):
    if filename.endswith(".json"):
        file_path = os.path.join(folder_path, filename)
        tokens = parse_json(file_path)
        print(f"\nFile: {filename}")
        print("\n".join(tokens))
