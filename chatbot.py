import torch
import torch.nn as nn
import re
import tomllib

# ----- 1. Load config -----
with open("database/cache/configModel.toml", "rb") as f:
    cfg = tomllib.load(f)["model"]

modelType = cfg["model_type"]
input_size = cfg["input_size"]
hidden_size = cfg["hidden_size"]
output_size = cfg["output_size"]

# ----- 2. Load model & vocab -----
model_state = torch.load(f"database/{modelType}/chat_model_{modelType}.pth")
vocab = torch.load(f"database/{modelType}/vocab_{modelType}.pth")
label_map = torch.load(f"database/{modelType}/label_map_{modelType}.pth")

# ----- 3. Model khớp với config -----
class NLPChat(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc2(self.relu(self.fc1(x)))
        return x

model = NLPChat(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

# ----- 4. Tokenize & BOW -----
def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.split()

def bow_vector(sentence):
    vec = torch.zeros(len(vocab))
    for w in tokenize(sentence):
        if w in vocab:
            vec[vocab[w]] = 1
    return vec

# ----- 5. Chat loop -----
print(" LeastAI sẵn sàng! Gõ 'exit' để thoát.")
while True:
    text = input("You: ").strip()
    if text.lower() == "exit":
        break
    vec = bow_vector(text)
    if vec.sum() == 0:
        print("LeastAI: Tôi không hiểu…")
        continue
    with torch.no_grad():
        out = model(vec)
        label = torch.argmax(out).item()
        print("LeastAI:", label_map[label])
