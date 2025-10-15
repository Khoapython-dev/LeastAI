import json, pandas as pd

path = "/bootstrap/LeastAI/xp.csv"

with open(path) as f:
    data = json.load(f)

rows = []
for k, v in data.items():
    info = v.get("original dialog info", {})
    rows.append({
        "prompt": info.get("literal", ""),
        "response": info.get("narrative", "")
    })

df = pd.DataFrame(rows)
df["response"] = df["response"].str.replace("AI", "LeastAI", case=False)
df.to_csv("leastai_dataset.csv", index=False)
print("✅ Done, saved to leastai_dataset.csv")
