import json

def create_keyword_dictionary(vocab_file, output_file, selected_fields=None):
    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    dictionary = {}
    counter = 1

    # Nếu không chỉ định fields nào, lấy tất cả
    if selected_fields is None:
        selected_fields = vocab.keys()

    for field in selected_fields:
        for keyword in vocab.get(field, []):
            if keyword not in dictionary:
                dictionary[keyword] = counter
                counter += 1

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dictionary, f, ensure_ascii=False, indent=4)
    
    print(f"[+] Dictionary đã lưu tại: {output_file}")

# ======== Sử dụng =========

vocab_path = "vocab.json"
dict_output_path = "dictionary.json"

# Chọn trường cần lấy từ vocab
selected_fields = ["key", "value"]  # ← có thể là ["attribute", "tag"] hoặc để None để lấy tất cả

create_keyword_dictionary(vocab_path, dict_output_path, selected_fields)
