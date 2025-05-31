import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.nn.utils.rnn import pad_sequence
import math
from torch.optim import AdamW

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

with open("vocab_mysql.json") as f:
    vocab = json.load(f)

def extract_all_ints(d):
    ints = []
    if isinstance(d, dict):
        for v in d.values():
            ints.extend(extract_all_ints(v))
    elif isinstance(d, int):
        ints.append(d)
    return ints

all_indices = extract_all_ints(vocab)
vocab_size = max(all_indices) + 1
vocab_size += 00
print(f"Vocabulary size: {vocab_size}")

class ConfigIssueDataset(Dataset):
    def __init__(self, benign_dir, issue_dir):
        self.samples = []
        self.labels = []
        self._load_files(benign_dir, 0)
        self._load_files(issue_dir, 1)

    def _load_files(self, directory, label):
        count = 0
        for file in os.listdir(directory):
            if count >= 400:  # Giới hạn số lượng file
                break
            path = os.path.join(directory, file)
            with open(path) as f:
                tokens = json.load(f)
                self.samples.append(tokens)
                self.labels.append(label)
            count += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx], dtype=torch.long), torch.tensor(self.labels[idx])

# Collate function để padding batch
def collate_batch(batch):
    xs, ys = zip(*batch)
    xs_pad = pad_sequence(xs, batch_first=True, padding_value=0)
    ys = torch.stack(ys)
    return xs_pad, ys

# class ClassifierModel(nn.Module): 
#     def __init__(self, vocab_size, embed_dim=512, num_heads=8, num_classes=2, dropout=0.3):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, embed_dim)
#         self.multihead = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
#         self.dropout = nn.Dropout(dropout)

#         # Fully-connected layers
#         self.fc_layers = nn.Sequential(
#             nn.Linear(embed_dim, 128),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(128, num_classes)
#         )

#     def forward(self, x):
#         # x: (batch_size, seq_len)
#         x = self.embedding(x)  # (batch_size, seq_len, embed_dim)
#         attn_output, _ = self.multihead(x, x, x)  # (batch_size, seq_len, embed_dim)
#         attn_output = self.dropout(attn_output)

#         # Tính trung bình theo chiều seq_len (mean pooling)
#         x_avg = attn_output.mean(dim=1)  # (batch_size, embed_dim)

#         # Đưa vào fully connected
#         out = self.fc_layers(x_avg)
#         return out


# fix qua fullyconnected layer
# coding seft attention bằng công thức

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Q, K, V projection
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # LayerNorm
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None, return_attention=False):
        # x: (batch, seq_len, embed_dim)
        batch_size, seq_len, _ = x.size()

        # Linear projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to (batch, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (batch, heads, seq_len, seq_len)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len) for broadcasting
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)  # (batch, heads, seq_len, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        # Output projection + dropout
        out = self.out_proj(attn_output)
        out = self.resid_dropout(out)

        # Add & Norm
        out = self.ln(x + out)

        if return_attention:
            return out, attn_weights
        else:
            return out


class ClassifierModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, num_heads=8, num_classes=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attn = MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.fc_layers = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.attn(x, mask=mask)
        x = x.mean(dim=1)
        return self.fc_layers(x)
    #có thể thêm lớp chuẩn hóa trước khi qua lớp fully connected, scale lại trong miền data output frame 
    # scale về miền [0, 1] hoặc [-1, 1] nếu cần thiết



def plot_confusion_matrix(y_true, y_pred, filename="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign", "Issue"], yticklabels=["Benign", "Issue"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Confusion matrix saved to {filename}")
    plt.close()

def evaluate(model, dataloader, criterion, threshold=0.5):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item()
            probs = F.softmax(out, dim=1)
            probs_class_1 = probs[:, 1]
            preds = (probs_class_1 > threshold).long()
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    print(f"Confusion Matrix (threshold={threshold}):")
    print(confusion_matrix(all_labels, all_preds))
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, digits=4))
    plot_confusion_matrix(all_labels, all_preds)
    return total_loss / len(dataloader), acc

def find_best_threshold(model, dataloader):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            probs = F.softmax(out, dim=1)[:, 1]
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    best_thresh = 0.0
    best_f1 = 0.0
    for thresh in [i * 0.01 for i in range(10, 90)]:
        preds = [1 if p > thresh else 0 for p in all_probs]
        f1 = f1_score(all_labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    print(f"Best threshold: {best_thresh:.2f} with F1 score: {best_f1:.4f}")
    return best_thresh

def train():
    dataset = ConfigIssueDataset("mysql_benign_tokens", "mysql_issue_tokens")
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=16, collate_fn=collate_batch)

    model = ClassifierModel(vocab_size).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    best_model_path = "best_model.pth"

    for epoch in range(1, 101):
        model.train()
        train_loss = 0
        train_preds, train_labels = [], []

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = torch.argmax(out, dim=1)
            train_preds.extend(preds.cpu().tolist())
            train_labels.extend(y.cpu().tolist())

        avg_train_loss = train_loss / len(train_loader)
        train_acc = accuracy_score(train_labels, train_preds)

        val_loss, val_acc = evaluate(model, val_loader, criterion, threshold=0.5)

        print(f"\nEpoch {epoch}")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train Acc: {train_acc:.4f}   | Val Acc: {val_acc:.4f}")
        print("-" * 50)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

    print("\nTraining finished.")
    print(f"Best val acc: {best_val_acc:.4f}. Model saved to {best_model_path}")

    # Load best model
    model.load_state_dict(torch.load(best_model_path))

    # Tìm threshold tốt nhất trên validation
    best_threshold = find_best_threshold(model, val_loader)

    # Đánh giá lại với threshold tối ưu
    print("\n=== Final evaluation with best threshold ===")
    evaluate(model, val_loader, criterion, threshold=best_threshold)

    print("\nModel structure:")
    print(model)

if __name__ == "__main__":
    train()
