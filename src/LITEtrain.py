# src/SUtrain_Lite.py
import os, re, torch, tomllib, tomli_w
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from colorama import Fore

# ========== 1. Đọc loại model ==========
with open("model/type/type.toml", "rb") as f:
    t = tomllib.load(f)
modelType = str(t["train"]["model_type"])

# ========== 2. Load dữ liệu ==========
data_path = f"model/{modelType}.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {data_path}")

df = pd.read_csv(data_path)
df["prompt"] = df.get("prompt", "").fillna("").astype(str)
df["response"] = df.get("response", "").fillna("").astype(str)
df = df[(df["prompt"] != "") & (df["response"] != "")]
if df.empty:
    raise ValueError("❌ Dữ liệu trống — kiểm tra lại file CSV.")

# Giới hạn nhẹ
df = df.sample(n=min(len(df), 5000), random_state=42)

inputs = df["prompt"].tolist()
outputs = df["response"].tolist()

# ========== 3. Tokenizer siêu nhẹ ==========
def tokenize(txt: str):
    txt = txt.lower()
    txt = re.sub(r"[^a-zA-Z0-9\sÀ-ỹ]", " ", txt)
    return txt.split()

vocab = sorted(set(w for s in (inputs + outputs) for w in tokenize(s)))
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}

if not vocab:
    raise ValueError("❌ Không tạo được từ vựng — kiểm tra dữ liệu.")

def vectorize(sentence):
    vec = torch.zeros(len(vocab))
    for w in tokenize(sentence):
        if w in word2idx:
            vec[word2idx[w]] = 1
    return vec

# ========== 4. Model TinyChat-Lite ==========
class TinyChatLite(nn.Module):
    def __init__(self, vocab_size, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(vocab_size, hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)

model = TinyChatLite(len(vocab))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ========== 5. Resume checkpoint ==========
ckpt_path = f"database/{modelType}/TinyChatLite_ckpt.pth"
if os.path.exists(ckpt_path):
    print(Fore.YELLOW + "⚙️ Resume từ checkpoint cũ...")
    model.load_state_dict(torch.load(ckpt_path))

# ========== 6. Train ==========
epochs = 1000
batch_size = 8
print(Fore.CYAN + f"🚀 Training TinyChat-Lite ({modelType}) — CPU tối ưu, RAM tiết kiệm")

def make_context(p, r):
    return f"{p.strip()} {r.strip()}"

try:
    for epoch in range(epochs):
        total_loss = 0.0
        for i in range(0, len(inputs), batch_size):
            batch_in = inputs[i:i+batch_size]
            batch_out = outputs[i:i+batch_size]

            X_batch = torch.stack([vectorize(p) for p in batch_in])
            Y_batch = torch.stack([vectorize(r) for r in batch_out])

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, torch.argmax(Y_batch, dim=1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 20 == 0:
            avg_loss = total_loss / max(1, len(inputs) / batch_size)
            print(f"{Fore.GREEN}Epoch {epoch:04d} | Loss: {avg_loss:.4f}")

        if epoch % 100 == 0 and epoch > 0:
            os.makedirs(f"database/{modelType}", exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
            print(Fore.YELLOW + f"💾 Checkpoint saved at epoch {epoch}")

except KeyboardInterrupt:
    print(Fore.RED + "\n🟥 Dừng thủ công — lưu checkpoint...")
    os.makedirs(f"database/{modelType}", exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)

# ========== 7. Lưu model & config ==========
torch.save(model.state_dict(), f"database/{modelType}/TinyChatLite_{modelType}.pth")
torch.save(word2idx, f"database/{modelType}/vocab_{modelType}.pth")
torch.save(idx2word, f"database/{modelType}/vocab_reverse_{modelType}.pth")

config = {
    "model": {
        "vocab_size": len(vocab),
        "hidden_size": 128,
        "type": modelType,
        "lite": True
    }
}
os.makedirs("database/cache", exist_ok=True)
with open("database/cache/configModelLite.toml", "wb") as f:
    tomli_w.dump(config, f)

print(Fore.CYAN + f"✅ Training hoàn tất! Model TinyChat-Lite lưu tại: database/{modelType}/")
