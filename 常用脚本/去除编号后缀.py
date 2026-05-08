"""递归去除文件和文件夹名称结尾的（编号/字母）后缀。"""
from pathlib import Path
import re

# ======== 配置 ========
TARGET_FOLDER = Path(r"E:\DATA\Project\OCR-ASR-with-MLLM\摩诃止观-定智法师 2017")
# =====================

pattern = re.compile(r"[（(][\dA-Za-z]+[）)]$")  # 匹配结尾的（1）(A)（123b）等

for item in sorted(TARGET_FOLDER.rglob("*"), reverse=True):
    stem = item.stem or item.name  # 文件取 stem，文件夹取 name
    new_stem = pattern.sub("", stem)
    if new_stem == stem:
        continue  # 没匹配到，跳过

    new_name = new_stem + item.suffix  # 文件有后缀，文件夹 suffix 为空
    new_path = item.parent / new_name
    if new_path.exists():
        print(f"  {item.relative_to(TARGET_FOLDER)}  →  跳过（{new_path.name} 已存在）")
        continue
    item.rename(new_path)
    print(f"  {item.relative_to(TARGET_FOLDER)}  →  {new_path.name}")
