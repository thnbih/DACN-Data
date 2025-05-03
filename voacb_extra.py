import os
import xml.etree.ElementTree as ET
from collections import defaultdict
import json
import re
import yaml

def extract_from_xml(xml_file):
    syntax_dict = defaultdict(set)
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        def recurse(element):
            syntax_dict["tag"].add(element.tag)
            for attr in element.attrib:
                syntax_dict["attribute"].add(attr)
                syntax_dict["value"].add(element.attrib[attr])
            for child in element:
                recurse(child)

        recurse(root)
    except ET.ParseError:
        print(f"[!] Lỗi cú pháp XML: {xml_file}")
    return syntax_dict

def extract_from_conf(file_path):
    syntax_dict = defaultdict(set)
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        raw = f.read()
        lines = normalize_lines(raw, file_path).splitlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line.strip("[]").strip()
            syntax_dict["section"].add(section)
        elif "=" in line:
            key, val = map(str.strip, line.split("=", 1))
            if key: syntax_dict["key"].add(key)
            if val: syntax_dict["value"].add(val)
        else:
            words = re.findall(r'\b\w+\b', line)
            for word in words:
                syntax_dict["flag"].add(word)
    return syntax_dict

def extract_from_yaml(file_path):
    syntax_dict = defaultdict(set)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw = f.read()
            cleaned = normalize_lines(raw, file_path)
            data = yaml.safe_load(cleaned)

        def recurse(d):
            if isinstance(d, dict):
                for k, v in d.items():
                    syntax_dict["key"].add(str(k))
                    recurse(v)
            elif isinstance(d, list):
                for item in d:
                    recurse(item)
            else:
                syntax_dict["value"].add(str(d))
        recurse(data)
    except Exception as e:
        print(f"[!] Lỗi YAML: {file_path} ({e})")
    return syntax_dict

def extract_from_json(file_path):
    syntax_dict = defaultdict(set)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw = f.read()
            cleaned = normalize_lines(raw, file_path)
            data = json.loads(cleaned)

        def recurse(d):
            if isinstance(d, dict):
                for k, v in d.items():
                    syntax_dict["key"].add(str(k))
                    recurse(v)
            elif isinstance(d, list):
                for item in d:
                    recurse(item)
            else:
                syntax_dict["value"].add(str(d))
        recurse(data)
    except Exception as e:
        print(f"[!] Lỗi JSON: {file_path} ({e})")
    return syntax_dict

def merge_dicts(main_dict, new_dict):
    for key in new_dict:
        main_dict[key].update(new_dict[key])

def normalize_lines(raw_text, file_path):
    ext = file_path.lower()

    # Tách riêng phần thân sau các thẻ mở dạng <IfModule ...>
    raw_text = re.sub(r'(</?\w+[^>]*>)([^\n<])', r'\1\n\2', raw_text)

    # Với các directive bị dính: chèn newline giữa các directive liên tiếp
    raw_text = re.sub(r'(?<=[a-zA-Z0-9])(\s{2,})([A-Za-z_]+\s+\S)', r'\n\2', raw_text)

    if ext.endswith(".xml"):
        return raw_text.replace("><", ">\n<")
    elif ext.endswith((".json",)):
        return raw_text.replace("},", "},\n").replace("],", "],\n")
    elif ext.endswith((".yaml", ".yml")):
        return raw_text
    elif ext.endswith((".conf", ".cnf", ".ini")):
        raw_text = re.sub(r'(;|\})', r'\1\n', raw_text)
        return raw_text
    else:
        return raw_text



def build_vocab_from_folder(folder_path):
    final_vocab = defaultdict(set)
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        ext = filename.lower()
        if ext.endswith(".xml"):
            syntax = extract_from_xml(filepath)
        elif ext.endswith((".conf", ".cnf", ".ini")):
            syntax = extract_from_conf(filepath)
        elif ext.endswith((".yaml", ".yml")):
            syntax = extract_from_yaml(filepath)
        elif ext.endswith(".json"):
            syntax = extract_from_json(filepath)
        else:
            continue
        merge_dicts(final_vocab, syntax)
    return final_vocab

def save_vocab_to_json(vocab, output_file):
    vocab_serializable = {k: sorted(list(v)) for k, v in vocab.items()}
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(vocab_serializable, f, ensure_ascii=False, indent=4)
    print(f"[+] Đã lưu vocab vào {output_file}")

def highlight_keywords_in_file(file_path, vocab):
    highlights = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            raw = f.read()
            ext = file_path.lower()
            clean_text = normalize_lines(raw, ext)
            lines = clean_text.splitlines()
            for idx, line in enumerate(lines):
                for keyword in set().union(*vocab.values()):
                    if keyword in line:
                        highlights.append((idx + 1, line.strip()))
                        break
    except Exception as e:
        print(f"[!] Lỗi đọc file {file_path}: {e}")
    return highlights

# ==== Sử dụng ====
folder_path = "F:\Study\DACN\dataset\configissues\httpd"  # ← Thay bằng đường dẫn thật
output_file = "vocab_httpd.json"

vocab = build_vocab_from_folder(folder_path)
save_vocab_to_json(vocab, output_file)

print("\n[+] Vocabulary thu được:")
for k, v in vocab.items():
    print(f"{k}: {sorted(v)}")

print("\n[+] Đánh dấu các dòng có keyword trong từng file:")
for filename in os.listdir(folder_path):
    if filename.endswith((".xml", ".conf", ".cnf", ".ini", ".yaml", ".yml", ".json")):
        file_path = os.path.join(folder_path, filename)
        highlighted = highlight_keywords_in_file(file_path, vocab)
        print(f"\nFile: {filename}")
        for line_num, content in highlighted:
            print(f"  Dòng {line_num}: {content}")
