import os
import xml.etree.ElementTree as ET
from collections import defaultdict
import json

def extract_syntax_keywords(xml_file):
    syntax_dict = defaultdict(set)
    with open(xml_file, 'r', encoding='utf-8') as file:
        try:
            tree = ET.parse(file)
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
            print(f"[!] Lỗi cú pháp: {xml_file}")
    return syntax_dict

def merge_dicts(main_dict, new_dict):
    for key in new_dict:
        main_dict[key].update(new_dict[key])

def build_vocab_from_folder(folder_path):
    final_vocab = defaultdict(set)
    for filename in os.listdir(folder_path):
        if filename.endswith(".xml"):
            filepath = os.path.join(folder_path, filename)
            syntax = extract_syntax_keywords(filepath)
            merge_dicts(final_vocab, syntax)
    return final_vocab

def save_vocab_to_json(vocab, output_file):
    # Chuyển set sang list để ghi vào JSON
    vocab_serializable = {k: sorted(list(v)) for k, v in vocab.items()}
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(vocab_serializable, f, ensure_ascii=False, indent=4)
    print(f"[+] Đã lưu vocab vào {output_file}")

# ==== Sử dụng ====
folder_path = "F:\Study\DACN\dataset\configissues\hadoop"  # <-- thay bằng thư mục của bạn
output_file = "vocab2.json"

vocab = build_vocab_from_folder(folder_path)
save_vocab_to_json(vocab, output_file)

# import os
# import xml.etree.ElementTree as ET
# from collections import defaultdict

# def extract_syntax_keywords(xml_file):
#     syntax_dict = defaultdict(set)
#     with open(xml_file, 'r', encoding='utf-8') as file:
#         try:
#             tree = ET.parse(file)
#             root = tree.getroot()

#             def recurse(element):
#                 syntax_dict["tag"].add(element.tag)
#                 for attr in element.attrib:
#                     syntax_dict["attribute"].add(attr)
#                     syntax_dict["value"].add(element.attrib[attr])
#                 for child in element:
#                     recurse(child)

#             recurse(root)
#         except ET.ParseError:
#             print(f"[!] Lỗi cú pháp: {xml_file}")
#     return syntax_dict

# def merge_dicts(main_dict, new_dict):
#     for key in new_dict:
#         main_dict[key].update(new_dict[key])

# def build_vocab_from_folder(folder_path):
#     final_vocab = defaultdict(set)
#     for filename in os.listdir(folder_path):
#         if filename.endswith(".xml"):
#             filepath = os.path.join(folder_path, filename)
#             syntax = extract_syntax_keywords(filepath)
#             merge_dicts(final_vocab, syntax)
#     return final_vocab

# def highlight_keywords_in_file(file_path, vocab):
#     highlights = []
#     with open(file_path, 'r', encoding='utf-8') as f:
#         lines = f.readlines()
#         for idx, line in enumerate(lines):
#             for keyword in vocab["tag"].union(vocab["attribute"]).union(vocab["value"]):
#                 if keyword in line:
#                     highlights.append((idx + 1, line.strip()))
#                     break
#     return highlights

# # ==== Sử dụng ====
# folder_path = "duong_dan_toi_folder_xml"  # <-- thay bằng thư mục của bạn

# vocab = build_vocab_from_folder(folder_path)
# print("\n[+] Vocabulary thu được:")
# for k, v in vocab.items():
#     print(f"{k}: {sorted(v)}")

# print("\n[+] Đánh dấu các dòng có keyword trong từng file:")
# for filename in os.listdir(folder_path):
#     if filename.endswith(".xml"):
#         file_path = os.path.join(folder_path, filename)
#         highlighted = highlight_keywords_in_file(file_path, vocab)
#         print(f"\nFile: {filename}")
#         for line_num, content in highlighted:
#             print(f"  Dòng {line_num}: {content}")
