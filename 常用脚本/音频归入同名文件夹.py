"""将目标文件夹下所有音频文件移入各自的同名文件夹。"""
from pathlib import Path
import shutil

# ======== 配置 ========
TARGET_FOLDER = Path(r"E:\DATA\Project\OCR-ASR-with-MLLM\摩诃止观-定智法师 2017")
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".wma"}
# =====================

for file in sorted(TARGET_FOLDER.rglob("*"), reverse=True):
    if not file.is_file() or file.suffix.lower() not in AUDIO_EXTENSIONS:
        continue

    new_dir = file.parent / file.stem
    new_dir.mkdir(exist_ok=True)
    new_path = new_dir / file.name
    shutil.move(str(file), str(new_path))
    print(f"  {file.relative_to(TARGET_FOLDER)}  →  {new_path.relative_to(TARGET_FOLDER)}")

print("\n完成。")
