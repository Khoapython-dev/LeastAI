import torch
import torch.nn as nn
import torch.optim as optim
from colorama import Fore
import pandas as pd
import re

# ----- 1. Load dữ liệu -----
df = pd.read_csv("chat_data_max.csv")
df['prompt'] = df['prompt'].fillna("").astype(str)
df['response'] = df['response'].fillna("").astype(str)

inputs = df['prompt'].tolist()
outputs = df['response'].tolist()

# ----- 2. Tokenize -----
def tokenize(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.split()

all_words = []
for sentence in inputs:
    all_words.extend(tokenize(sentence))

vocab = {word: i for i, word in enumerate(set(all_words))}
label_map = {i: out for i, out in enumerate(outputs)}

# ----- 3. Bag-of-Words vector (on-the-fly) -----
def bow_vector(sentence):
    vec = torch.zeros(len(vocab))
    for w in tokenize(sentence):
        if w in vocab:
            vec[vocab[w]] = 1
    return vec

# ----- 4. Model nhẹ -----
class NLPChat(nn.Module):
    def __init__(self, input_size, embed_size=500, hidden_size=500, output_size=None):
        super().__init__()
        self.embed = nn.Linear(input_size, embed_size)
        self.fc1 = nn.Linear(embed_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = self.embed(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = NLPChat(input_size=len(vocab), output_size=len(outputs))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# ----- 5. Mini-batch training -----
batch_size = 500  # nhỏ để RAM nhẹ
num_epochs = 5000

print("Training model (mini-batch, on-the-fly)... Nhấn Ctrl+C để tạm dừng và save model nếu muốn.")
for epoch in range(num_epochs):
    perm = torch.randperm(len(inputs))
    for i in range(0, len(inputs), batch_size):
        idx = perm[i:i+batch_size]
        X_batch = torch.stack([bow_vector(inputs[j]) for j in idx])
        y_batch = torch.tensor(idx, dtype=torch.long)

        optimizer.zero_grad()
        out = model(X_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        optimizer.step()
    
    if epoch % 1 == 0:
        print(f"{Fore.GREEN}Epoch {epoch}, Loss: {loss.item():.4f}")

# ----- 6. Lưu model & vocab -----
torch.save(model.state_dict(), "chat_model_max_light.pth")
torch.save(vocab, "vocab_max_light.pth")
torch.save(label_map, "label_map_max_light.pth")
print("Training xong, model và vocab đã lưu!")
