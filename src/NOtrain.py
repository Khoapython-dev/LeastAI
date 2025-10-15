# src/traint.py
import os, re, torch, tomllib, tomli_w
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from colorama import Fore

# ----- 1. Đọc loại model -----
with open("model/type/type.toml", "rb") as f:
    t = tomllib.load(f)
modelType = str(t["train"]["model_type"])

# ----- 2. Load dữ liệu -----
data_path = f"model/{modelType}.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {data_path}")

df = pd.read_csv(data_path)

if "prompt" not in df.columns:
    df["prompt"] = ""
if "response" not in df.columns:
    df["response"] = ""

df["prompt"] = df["prompt"].fillna("").astype(str)
df["response"] = df["response"].fillna("").astype(str)
df = df[(df["prompt"] != "") & (df["response"] != "")]  # bỏ dòng trống

if df.empty:
    raise ValueError("❌ Dữ liệu trống — kiểm tra lại file CSV trước khi train.")

inputs = df["prompt"].tolist()
outputs = df["response"].tolist()

# ----- 3. Tokenize -----
def tokenize(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.split()

vocab = {word: i for i, word in enumerate(sorted(set(w for s in inputs for w in tokenize(s))))}
if not vocab:
    raise ValueError("❌ Không tạo được vocab — có thể input chỉ chứa ký tự đặc biệt hoặc trống.")

label_map = {i: out for i, out in enumerate(outputs)}

def bow_vector(sentence):
    vec = torch.zeros(len(vocab))
    for w in tokenize(sentence):
        if w in vocab:
            vec[vocab[w]] = 1
    return vec

# ----- 4. Định nghĩa model -----
class TinyChat(nn.Module):
    def __init__(self, input_size, hidden=128, output_size=None):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, output_size)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

input_size = len(vocab)
output_size = len(outputs)
hidden_size = 128
model = TinyChat(input_size, hidden_size, output_size)

# ----- 5. Resume checkpoint -----
checkpoint_path = f"database/{modelType}/chat_checkpoint.pth"
if os.path.exists(checkpoint_path):
    print(Fore.YELLOW + "⚙️  Resume từ checkpoint cũ...")
    model.load_state_dict(torch.load(checkpoint_path))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

# ----- 6. Train -----
batch_size = 128
epochs = 10000

print(Fore.CYAN + f" Training TinyChat ({modelType})... Nhấn Ctrl+C để dừng.")
try:
    for epoch in range(epochs):
        if len(inputs) == 0:
            raise ValueError("Không có dữ liệu đầu vào để train.")
        perm = torch.randperm(len(inputs))
        total_loss = 0

        for i in range(0, len(inputs), batch_size):
            idx = perm[i:i + batch_size]
            X_batch = torch.stack([bow_vector(inputs[j]) for j in idx])
            y_batch = torch.tensor(idx, dtype=torch.long)

            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(inputs), 1)
        print(f"{Fore.GREEN}Epoch {epoch:04d} | Loss: {avg_loss:.4f}")

        if epoch % 50 == 0 and epoch > 0:
            os.makedirs(f"database/{modelType}", exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            print(Fore.YELLOW + f" 💾 Checkpoint saved at epoch {epoch}")

except KeyboardInterrupt:
    print(Fore.RED + "\n🟥 Dừng thủ công — lưu checkpoint...")
    os.makedirs(f"database/{modelType}", exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)

# ----- 7. Lưu model và mapping -----
os.makedirs(f"database/{modelType}", exist_ok=True)
torch.save(model.state_dict(), f"database/{modelType}/chat_model_{modelType}.pth")
torch.save(vocab, f"database/{modelType}/vocab_{modelType}.pth")
torch.save(label_map, f"database/{modelType}/label_map_{modelType}.pth")

# ----- 8. Ghi config -----
os.makedirs("database/cache", exist_ok=True)
config_data = {
    "model": {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "model_type": modelType,
    }
}
with open("database/cache/configModel.toml", "wb") as f:
    tomli_w.dump(config_data, f)

print(Fore.CYAN + f"✅ Training hoàn tất! Model và config lưu tại: database/{modelType}/")
