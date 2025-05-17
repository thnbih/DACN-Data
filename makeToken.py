import os
import json

def tokenize_keyword(text):
    tokens = []
    i = 0
    token = ""
    while i < len(text):
        if text[i] in ['/', '_']:
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

def generate_token_files_per_conf(folder_path, vocab_path, output_folder):
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(folder_path):
        if filename.endswith((".conf", ".cnf", ".ini")):
            file_path = os.path.join(folder_path, filename)
            current_section = None
            all_tokens = []

            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#') or line.startswith(';'):
                        continue

                    if line.startswith('[') and line.endswith(']'):
                        current_section = line[1:-1].strip()
                        continue

                    if '=' in line and current_section:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()

                        sec_idx = vocab["sections"].get(current_section)
                        if sec_idx is None:
                            continue

                        tokens = [sec_idx]  # giữ section index

                        key_tokens = tokenize_keyword(key)
                        for token in key_tokens:
                            tok_idx = vocab["section_keyword_tokens"].get(current_section, {}).get(token)
                            if tok_idx is not None:
                                tokens.append(tok_idx)

                        # nếu không tìm được token nào từ keyword -> bỏ dòng
                        if len(tokens) == 1:
                            continue

                        value_tokens = tokenize_keyword(value)
                        for token in value_tokens:
                            val_idx = vocab["value_tokens"].get(token)
                            if val_idx is not None:
                                tokens.append(val_idx)

                        for token in tokenize_keyword(value):
                            val_idx = vocab["value_tokens"].get(token)
                            if val_idx is not None:
                                tokens.append(val_idx)

                        all_tokens.extend(tokens)

            if all_tokens:
                out_filename = os.path.splitext(filename)[0] + ".json"
                out_path = os.path.join(output_folder, out_filename)
                with open(out_path, 'w', encoding='utf-8') as out_f:
                    json.dump(all_tokens, out_f)

    print(f"[+] Đã tạo các file token JSON tại: {output_folder}")
if __name__ == "__main__":
    folder_path = "configissues\\mysql"
    vocab_path = "vocab_mysql.json"
    output_folder = "output_mysql_issue_tokens"

    generate_token_files_per_conf(folder_path, vocab_path, output_folder)

