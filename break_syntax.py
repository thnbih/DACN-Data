import os
import xml.etree.ElementTree as ET
from collections import defaultdict
import json
import re
import yaml
import pandas as pd

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

def split_tokens(s):
    return re.split(r'(\W)', s)

def extract_keywords_by_section_conf(file_path):
    section_keywords = defaultdict(set)
    keyword_values = defaultdict(set)
    current_section = "global"

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        raw = f.read()
        lines = normalize_lines(raw, file_path).splitlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith("[") and line.endswith("]"):
            current_section = line.strip("[]").strip()
        elif "=" in line:
            key, val = map(str.strip, line.split("=", 1))
            for token in split_tokens(key):
                if token.strip():
                    section_keywords[current_section].add(token)
                    keyword_values[token].add(val)
            for token in split_tokens(val):
                if token.strip():
                    keyword_values[key].add(token)
        else:
            for word in split_tokens(line):
                if word.strip():
                    section_keywords[current_section].add(word)
    return section_keywords, keyword_values

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
    raw_text = re.sub(r'(</?\w+[^>]*>)([^\n<])', r'\1\n\2', raw_text)
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

def tokenize_keyword(text):
    tokens = []
    i = 0
    token = ""
    while i < len(text):
        if text[i] in ['/', '_', '\'', '\\']:
            if token:
                tokens.append(token)
                token = ""
            tokens.append(text[i])
        else:
            token += text[i]
        i += 1
    if token:
        tokens.append(token)
    return tokens

def expand_keywords_unique_ordered(section_keywords):
    expanded = {}
    for section, keywords in section_keywords.items():
        tokens = []
        for keyword in keywords:
            tokens.extend(tokenize_keyword(keyword))
        expanded[section] = list(dict.fromkeys(tokens))
    return expanded

def expand_values_unique_ordered(keyword_values):
    expanded = {}
    for keyword, values in keyword_values.items():
        tokens = []
        for value in values:
            tokens.extend(tokenize_keyword(value))
        expanded[keyword] = list(dict.fromkeys(tokens))
    return expanded

def save_conf_keywords_to_excel(section_keywords, keyword_values, output_excel):
    expanded_keywords = expand_keywords_unique_ordered(section_keywords)
    max_len_section = max(len(v) for v in expanded_keywords.values())
    for section in expanded_keywords:
        length = len(expanded_keywords[section])
        if length < max_len_section:
            expanded_keywords[section].extend([""] * (max_len_section - length))
    df_section = pd.DataFrame(expanded_keywords)
    expanded_values = expand_values_unique_ordered(keyword_values)
    max_len_value = max(len(v) for v in expanded_values.values())
    for keyword in expanded_values:
        length = len(expanded_values[keyword])
        if length < max_len_value:
            expanded_values[keyword].extend([""] * (max_len_value - length))
    df_keyword = pd.DataFrame(expanded_values)
    with pd.ExcelWriter(output_excel, engine='xlsxwriter') as writer:
        df_section.to_excel(writer, sheet_name='SectionKeywords', index=False)
        df_keyword.to_excel(writer, sheet_name='KeywordValues', index=False)

def save_conf_keywords_to_json_global_index(section_keywords, keyword_values, output_json):
    expanded_keywords = expand_keywords_unique_ordered(section_keywords)
    expanded_values = expand_values_unique_ordered(keyword_values)
    global_index = 0
    section_index = {}
    for section in sorted(section_keywords.keys()):
        section_index[section] = global_index
        global_index += 1
    section_keyword_index = {}
    for section in sorted(section_keywords.keys()):
        keywords = sorted(section_keywords[section])
        section_keyword_index[section] = {}
        for kw in keywords:
            section_keyword_index[section][kw] = global_index
            global_index += 1
    section_token_index = {}
    for section in sorted(expanded_keywords.keys()):
        tokens = expanded_keywords[section]
        section_token_index[section] = {}
        for token in tokens:
            if token not in section_token_index[section]:
                section_token_index[section][token] = global_index
                global_index += 1
    all_value_tokens = set()
    for tokens in expanded_values.values():
        all_value_tokens.update(tokens)
    value_index = {}
    for token in sorted(all_value_tokens):
        value_index[token] = global_index
        global_index += 1
    vocab = {
        "sections": section_index,
        "section_keyword_tokens": section_token_index,
        "value_tokens": value_index
    }
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)
    print(f"[+] Đã lưu vocab JSON với index toàn cục vào {output_json}")

folder_path = "mysql-all"
output_file = "vocab_mysql_full.json"
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
excel_output = "mysql_keywords_by_section.xlsx"
all_section_keywords = defaultdict(set)
all_keyword_values = defaultdict(set)
for filename in os.listdir(folder_path):
    if filename.endswith((".conf", ".cnf", ".ini")):
        file_path = os.path.join(folder_path, filename)
        sec_kw, kw_val = extract_keywords_by_section_conf(file_path)
        for sec, kws in sec_kw.items():
            all_section_keywords[sec].update(kws)
        for kw, vals in kw_val.items():
            all_keyword_values[kw].update(vals)
json_output = "vocab_mysql.json"
save_conf_keywords_to_excel(all_section_keywords, all_keyword_values, excel_output)
save_conf_keywords_to_json_global_index(all_section_keywords, all_keyword_values, json_output)