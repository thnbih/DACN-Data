import os
import json
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

def load_vocab(vocab_path):
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    return vocab  # dict: {token: id}

def tokenize_file(file_path, vocab, use_ids=False):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    tokens = word_tokenize(text)
    
    if use_ids:
        filtered = [str(vocab[token]) for token in tokens if token in vocab]
    else:
        filtered = [token for token in tokens if token in vocab]
    
    return ' '.join(filtered)

def process_folder(input_folder, vocab_path, output_folder, use_ids=False):
    os.makedirs(output_folder, exist_ok=True)
    vocab = load_vocab(vocab_path)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.conf', '.cnf', '.json')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            tokenized_text = tokenize_file(input_path, vocab, use_ids)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(tokenized_text)

# Ví dụ sử dụng:
input_folder = '.\\configissues\\all'
output_folder = '.\\token_issue_token'
vocab_file = 'vocab.json'
process_folder(input_folder, vocab_file, output_folder, use_ids=False)  # hoặc False nếu muốn token thay vì ID
