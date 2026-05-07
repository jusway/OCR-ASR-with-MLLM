"""将指定文件夹内除音频外的所有文件打包为 zip，保留目录结构。"""
from pathlib import Path
import zipfile

# ======== 配置 ========
SOURCE_FOLDER = Path(r"E:\DATA\Sync\ASR整理\摩诃止观-久仁法师")
OUTPUT_ZIP = SOURCE_FOLDER.parent / f"{SOURCE_FOLDER.name}.zip"
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".wma", ".wmv"}
# =====================

with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
    for file in SOURCE_FOLDER.rglob("*"):
        if file.is_file() and file.suffix.lower() not in AUDIO_EXTENSIONS:
            zf.write(file, file.relative_to(SOURCE_FOLDER))
            print(f"  + {file.relative_to(SOURCE_FOLDER)}")

print(f"\n打包完成：{OUTPUT_ZIP}")
