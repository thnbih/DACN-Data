import os
import json
import numpy as np
from gensim.models import Word2Vec

# Đường dẫn
token_folder = "token_issue_id"  # Thư mục chứa các file token
vocab_file = "vocab.json"  # File vocab JSON
output_folder = "vector_output_issue"  # Thư mục lưu vector
word2vec_model_path = "word2vec_issue.model"  # Đường dẫn đến mô hình word2vec

# Tạo thư mục output
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Đọc vocab từ file JSON
try:
    with open(vocab_file, 'r') as f:
        vocab = json.load(f)  # Từ điển ánh xạ token sang chỉ số
    # Tạo từ điển ngược: chỉ số sang token
    index_to_token = {str(v): k for k, v in vocab.items()}
    print(f"Đã đọc {len(vocab)} token trong vocab.")
except FileNotFoundError:
    print(f"Không tìm thấy file vocab: {vocab_file}")
    index_to_token = {}
    vocab = {}

# Đọc tất cả file token
token_sequences = []
filenames = []
for filename in os.listdir(token_folder):
    filepath = os.path.join(token_folder, filename)
    with open(filepath, 'r') as f:
        indices = f.read().strip().split()
        if not indices:
            #print(f"File {filename} rỗng, bỏ qua.")
            continue
        # Chuyển các chỉ số thành token
        tokens = []
        for idx in indices:
            token = index_to_token.get(idx, None)
            if token is None:
                #print(f"Chỉ số {idx} trong {filename} không có trong vocab.")
                tokens.append("<UNK>")  # Token không xác định
            else:
                tokens.append(token)
        token_sequences.append(tokens)
        filenames.append(filename)
print(f"Đã đọc {len(token_sequences)} file token.")

# Tải hoặc huấn luyện mô hình word2vec
try:
    model = Word2Vec.load(word2vec_model_path)
    print("Đã tải mô hình word2vec.")
except FileNotFoundError:
    if token_sequences:
        print("Huấn luyện mô hình word2vec mới...")
        model = Word2Vec(sentences=token_sequences, vector_size=256, window=5, min_count=1, workers=4)
        model.save(word2vec_model_path)
        print("Đã lưu mô hình word2vec.")
    else:
        raise ValueError("Không có dữ liệu token để huấn luyện word2vec.")

# Kiểm tra token có trong mô hình word2vec
for tokens, filename in zip(token_sequences, filenames):
    for token in tokens:
        if token not in model.wv:
            print(f"Token '{token}' trong {filename} không có trong mô hình word2vec.")

# Chuyển token thành vector và lưu
for tokens, filename in zip(token_sequences, filenames):
    vectors = []
    for token in tokens:
        if token in model.wv:
            vectors.append(model.wv[token])
        else:
            vectors.append(np.zeros(256))  # Vector cho token không xác định
            print(f"Token '{token}' trong {filename} không có vector, gán vector 0.")
    vectors = np.array(vectors)
    
    # Kiểm tra vector
    if np.all(vectors == 0):
        print(f"Cảnh báo: Vector cho {filename} chỉ chứa số 0.")
    else:
        print(f"Vector cho {filename} có {np.sum(vectors != 0)} giá trị khác 0.")

    # Lưu vector
    #output_filepath = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.npy")

    output_filepath = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.json")
    with open(output_filepath, 'w') as f:
        json.dump(vectors.tolist(), f)
        np.save(output_filepath, vectors)
    print(f"Đã lưu vector cho {filename} vào {output_filepath}")

print("Hoàn tất lưu vector vào thư mục", output_folder)