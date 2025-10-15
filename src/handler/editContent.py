import pandas as pd, re

df = pd.read_csv("xp.csv")

for col in df.columns:
    df[col] = df[col].astype(str).apply(lambda x: re.sub(r"\bAI\b", "leastAI", x, flags=re.IGNORECASE))

df.to_csv("l.csv", index=False)
print("✅ Đã cập nhật toàn bộ 'AI' thành 'leastAI' trong CSV")
