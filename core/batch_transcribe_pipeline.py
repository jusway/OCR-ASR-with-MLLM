"""
批量转录流水线：递归扫描文件夹，对每个音频文件调用 ASR 引擎生成逐字稿。
"""
import re
from pathlib import Path
from pydub import AudioSegment
from core.utils import Color


class BatchTranscriptionPipeline:
    """批量转录流水线：递归扫描文件夹，对每个音频文件调用 ASR 引擎生成逐字稿。"""

    def __init__(self, asr_engine, audio_extensions=None):
        self.asr_engine = asr_engine
        self.audio_extensions = audio_extensions or {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".wma"}

    def run(self, folder_path: str | Path):
        folder_path = Path(folder_path)
        audio_files = sorted(
            [f for f in folder_path.rglob("*") if f.is_file() and f.suffix.lower() in self.audio_extensions],
            key=lambda p: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(p))]
        )
        if not audio_files:
            print(f"{Color.RED}错误：在 {folder_path} 下没找到音频文件")
            exit(1)

        print(f"{Color.GREEN}找到 {len(audio_files)} 个音频文件")
        print(f"{Color.DARK_PURPLE}使用 {type(self.asr_engine).__name__} ASR 引擎")

        completed = 0
        total_seconds = 0.0

        for audio_path in audio_files:
            transcript_path = audio_path.parent / f"{audio_path.stem}_逐字稿.md"
            if transcript_path.exists():
                print(f"{Color.GREEN}[{audio_path.stem}] 逐字稿已存在，跳过")
                continue

            print(f"{Color.DARK_PURPLE}[{audio_path.stem}] 正在转录...")
            transcript_text = self.asr_engine.recognize(str(audio_path.resolve()))
            transcript_path.write_text(transcript_text, "utf-8")
            print(f"{Color.GREEN}[{audio_path.stem}] 逐字稿已保存")

            completed += 1
            duration_sec = len(AudioSegment.from_file(str(audio_path))) / 1000.0
            total_seconds += duration_sec
            total_minutes = total_seconds / 60.0
            print(f"{Color.GREEN}✅ 已完成第 {completed} 条音频转录（累计 {total_minutes:.1f} 分钟）")

        print(f"\n{Color.GREEN}全部完成。")