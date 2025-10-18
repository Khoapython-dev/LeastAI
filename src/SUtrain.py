# src/SUtrain.py
import os, torch, tomllib, tomli_w, pandas as pd
import sentencepiece as spm
import torch.nn as nn
import torch.optim as optim
from colorama import Fore

# ===== 1. Đọc loại model =====
with open("model/type/type.toml", "rb") as f:
    t = tomllib.load(f)
modelType = str(t["train"]["model_type"])

# ===== 2. Đọc dữ liệu =====
data_path = f"model/{modelType}.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {data_path}")

df = pd.read_csv(data_path)
df["prompt"] = df.get("prompt", "").fillna("").astype(str)
df["response"] = df.get("response", "").fillna("").astype(str)
df = df[(df["prompt"] != "") & (df["response"] != "")]
if df.empty:
    raise ValueError("❌ Dữ liệu trống — kiểm tra lại file CSV.")

inputs = df["prompt"].tolist()
outputs = df["response"].tolist()

# ===== 3. Tokenizer =====
os.makedirs("tokenizer", exist_ok=True)
sp_model_path = f"tokenizer/{modelType}_spm.model"

if not os.path.exists(sp_model_path):
    print(Fore.YELLOW + "🔤 Huấn luyện SentencePiece tokenizer (BPE)...")
    with open("tokenizer/tmp.txt", "w", encoding="utf-8") as f:
        for s in inputs + outputs:
            f.write(s.strip() + "\n")

    spm.SentencePieceTrainer.train(
        input="tokenizer/tmp.txt",
        model_prefix=f"tokenizer/{modelType}_spm",
        vocab_size=200000,  # Giảm bớt cho nhẹ
        character_coverage=1,
        model_type="bpe"
    )

sp = spm.SentencePieceProcessor(model_file=sp_model_path)
print(Fore.GREEN + "✅ Tokenizer loaded!")

def encode(text):
    return torch.tensor(sp.encode(text, out_type=int), dtype=torch.long)

def decode(ids):
    return sp.decode(ids.tolist())

# ===== 4. Model =====
class TinyChat(nn.Module):
    def __init__(self, vocab_size, hidden=2048):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.rnn = nn.GRU(hidden, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden

vocab_size = sp.vocab_size()
model = TinyChat(vocab_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.002)

# ===== 5. Resume checkpoint =====
ckpt_path = f"database/{modelType}/checkpoint.pth"
os.makedirs(f"database/{modelType}", exist_ok=True)
if os.path.exists(ckpt_path):
    print(Fore.YELLOW + "⚙️ Resume từ checkpoint...")
    model.load_state_dict(torch.load(ckpt_path))

# ===== 6. Hàm ghép prompt + response =====
def make_context(prompt, response):
    return f"[CONTEXT] {prompt.strip()} [REPLY] {response.strip()}"

# ===== 7. Train =====
epochs = 10000
print(Fore.CYAN + f"🚀 Training TinyChat_v2 ({modelType})...")

for epoch in range(epochs):
    total_loss = 0.0
    for p, r in zip(inputs, outputs):
        sentence = make_context(p, r)
        ids = encode(sentence)

        if len(ids) < 3:
            continue

        X = ids[:-1].unsqueeze(0)          # input
        y = ids[1:].unsqueeze(0)           # target
        optimizer.zero_grad()

        out, _ = model(X)
        out = out[:, :-1, :].reshape(-1, vocab_size)
        y = y[:, :out.shape[0]].reshape(-1)

        if out.shape[0] != y.shape[0]:
            continue  # tránh mismatch

        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / max(len(inputs), 1)
    print(f"{Fore.GREEN}Epoch {epoch:04d} | Loss: {avg_loss:.4f}")

    if epoch % 50 == 0 and epoch > 0:
        torch.save(model.state_dict(), ckpt_path)
        print(Fore.YELLOW + f"💾 Checkpoint saved — epoch {epoch}")

# ===== 8. Save model =====
torch.save(model.state_dict(), f"database/{modelType}/TinyChat_v2_{modelType}.pth")

config_data = {
    "model": {
        "vocab_size": vocab_size,
        "hidden_size": 2048,
        "type": modelType,
        "tokenizer": sp_model_path
    }
}
with open("database/cache/configModel.toml", "wb") as f:
    tomli_w.dump(config_data, f)

print(Fore.CYAN + f"✅ Training hoàn tất! Model TinyChat_v2 lưu tại: database/{modelType}/")
