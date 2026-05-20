"""统计指定文件夹下所有音频文件的个数（包括子文件夹）。"""
from pathlib import Path

# ============================================================
# 配置
# ============================================================
# folder_path = Path(r"D:\DATA\Sync\摩诃止观-定智法师 2017")
folder_path = Path(r"摩诃止观-定智法师 2017")
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".wma"}

# ============================================================
# 执行
# ============================================================
if __name__ == "__main__":
    if not folder_path.exists():
        print(f"❌ 文件夹不存在: {folder_path}")
        exit(1)

    audio_files = [f for f in folder_path.rglob("*") if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS]
    print(f"📂 {folder_path}")
    if audio_files:
        for f in audio_files:
            print(f"  {f}")
    print(f"\n🎵 音频文件总数: {len(audio_files)}")
