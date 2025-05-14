import os
import xml.etree.ElementTree as ET
from collections import defaultdict
import json
import re

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
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        # [section]
        if line.startswith("[") and line.endswith("]"):
            section = line.strip("[]").strip()
            syntax_dict["section"].add(section)
        # key=value
        elif "=" in line:
            key, val = map(str.strip, line.split("=", 1))
            if key: syntax_dict["key"].add(key)
            if val: syntax_dict["value"].add(val)
        # single-word flags
        else:
            words = re.findall(r'\b\w+\b', line)
            for word in words:
                syntax_dict["flag"].add(word)
    return syntax_dict

def merge_dicts(main_dict, new_dict):
    for key in new_dict:
        main_dict[key].update(new_dict[key])

def build_vocab_from_folder(folder_path):
    final_vocab = defaultdict(set)
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if filename.endswith(".xml"):
            syntax = extract_from_xml(filepath)
        elif filename.endswith((".conf", ".cnf")):
            syntax = extract_from_conf(filepath)
        else:
            continue
        merge_dicts(final_vocab, syntax)
    return final_vocab

def save_vocab_to_json(vocab, output_file):
    # Chuyển set sang list để lưu JSON
    vocab_serializable = {k: sorted(list(v)) for k, v in vocab.items()}
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(vocab_serializable, f, ensure_ascii=False, indent=4)
    print(f"[+] Đã lưu vocab vào {output_file}")

# ==== Sử dụng ====
folder_path = "F:\Study\DACN\dataset\configissues\mysql" 
output_file = "vocab_mysql.json"

vocab = build_vocab_from_folder(folder_path)
save_vocab_to_json(vocab, output_file)
