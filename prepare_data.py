import pandas as pd
import json

df = pd.read_excel("comments.xlsx")  

data = {}

for idx, row in df.iterrows():
    comment_key = f"comment{idx+1}"
    post_id = str(row["id"]).strip()
    comment_raw = str(row["comment"])
    comment_lines = [ " ".join(line.split()) for line in comment_raw.splitlines() ]
    comment_text = "\n".join(comment_lines)

    image_url = f"https://huggingface.co/datasets/yuu1234/850_comments/resolve/main/{post_id}_original.jpg"

    data[comment_key] = {
        "id": post_id,
        "comment": comment_text,
        "image_url": image_url
    }

with open("comments.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print("Đã tạo xong comments.json (comment đã chuẩn hóa)")
