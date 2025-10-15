import pandas as pd
import re

df = pd.read_csv("model/llmt-1.5.csv")
prompts, responses = [], []

for row in df['conversations']:
    try:
        # Lấy tất cả 'from':'human' và 'from':'gpt' cùng giá trị
        human_vals = re.findall(r"{'from': 'human', 'value': '(.*?)'}", row)
        gpt_vals = re.findall(r"{'from': 'gpt', 'value': '(.*?)'}", row)
        
        # Ghép theo cặp
        for h, g in zip(human_vals, gpt_vals):
            prompts.append(h.replace('""', '"'))
            responses.append(g.replace('""', '"').replace("AI", "LeastAI"))
    except Exception as e:
        print("Error parsing row:", e)

df_new = pd.DataFrame({"prompt": prompts, "response": responses})
df_new.to_csv("model/llmt-1.5-processed.csv", index=False)
print("✅ File processed lưu tại: model/llmt-1.5-processed.csv")
