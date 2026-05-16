"""递归批量转录文件夹下所有音频文件。"""
import re
from pathlib import Path

from pydub import AudioSegment
from core.asr_engine import Qwen3ASRFlashFiletrans
from core.utils import Color

# ============================================================
# ================ 配置 =======================================
# ============================================================

folder_path = Path(r"摩诃止观-定智法师 2017")
asr_model_name = "qwen3-asr-flash-filetrans"
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".wma"}

# ============================================================
# ================ 执行 =======================================
# ============================================================
if __name__ == "__main__":
    audio_files = sorted(
        [f for f in folder_path.rglob("*") if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS],
        key=lambda p: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(p))]
    )
    if not audio_files:
        print(f"{Color.RED}错误：在 {folder_path} 下没找到音频文件")
        exit(1)

    print(f"{Color.GREEN}找到 {len(audio_files)} 个音频文件")
    asr_engine = Qwen3ASRFlashFiletrans(model_name=asr_model_name)

    completed = 0
    total_seconds = 0.0

    for audio_path in audio_files:
        transcript_path = audio_path.parent / f"{audio_path.stem}_逐字稿.md"
        if transcript_path.exists():
            print(f"{Color.GREEN}[{audio_path.stem}] 逐字稿已存在，跳过")
            continue

        print(f"{Color.DARK_PURPLE}[{audio_path.stem}] 正在转录...")
        transcript_text = asr_engine.recognize(str(audio_path))
        transcript_path.write_text(transcript_text, "utf-8")
        print(f"{Color.GREEN}[{audio_path.stem}] 逐字稿已保存")

        completed += 1
        duration_sec = len(AudioSegment.from_file(str(audio_path))) / 1000.0
        total_seconds += duration_sec
        total_minutes = total_seconds / 60.0
        print(f"{Color.CYAN}✅ 已完成第 {completed} 条音频转录（累计 {total_minutes:.1f} 分钟）")

    print(f"\n{Color.GREEN}全部完成。")
