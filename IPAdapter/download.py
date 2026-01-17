# from datasets import load_dataset
# from PIL import Image
# import os, json

# out_root = "./dataset/flickr8k"
# os.makedirs(f"{out_root}/images", exist_ok=True)

# ds = load_dataset("Naveengo/flickr8k", split="train")  # 8.09k ảnh, cột image+text
# records = []
# for i, ex in enumerate(ds):
#     img: Image.Image = ex["image"]
#     caption = ex.get("text", "") or ""
#     rel = f"images/{i:06d}.jpg"
#     img.convert("RGB").save(os.path.join(out_root, rel), quality=95)
#     records.append({"image_file": rel, "text": caption})

# with open(os.path.join(out_root, "train.json"), "w", encoding="utf-8") as f:
#     json.dump(records, f, ensure_ascii=False, indent=2)

# print("Done:", len(records), "samples")
from datasets import load_dataset
from PIL import Image
import os, json, random

out_root = "./dataset/coco2014"
os.makedirs(f"{out_root}/images", exist_ok=True)

# Config "2014_captions" gom đủ 5 câu cho mỗi ảnh
ds = load_dataset("HuggingFaceM4/COCO", "2014_captions", split="train")

records = []
for i, ex in enumerate(ds):
    img: Image.Image = ex["image"]
    caps = ex.get("sentences_raw")  # list 5 captions
    if not caps:  # dự phòng cho config khác
        caps = [ex["sentences"]["raw"]]
    caption = random.choice(caps).strip()

    rel = f"images/{i:06d}.jpg"
    img.convert("RGB").save(os.path.join(out_root, rel), quality=95)
    records.append({"image_file": rel, "text": caption})

with open(os.path.join(out_root, "train.json"), "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print("Done:", len(records), "samples")


