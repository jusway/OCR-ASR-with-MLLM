"""递归批量转录文件夹下所有音频文件。"""
from pathlib import Path

from core.asr_engine import Qwen3ASRFlashFiletrans, MiMoLocalASR
from core.pipeline import BatchTranscriptionPipeline

# ============================================================
# ================ 配置 =======================================
# ============================================================

folder_path = Path(r"摩诃止观-定智法师 2017")

AVAILABLE_ASR = {
    "qwen": (Qwen3ASRFlashFiletrans, {"model_name": "qwen3-asr-flash-filetrans"}),
    "mimo-local": (MiMoLocalASR, {"base_url": "http://127.0.0.1:8000", "audio_tag": "<chinese>"}),
}
SELECTED_ASR = "mimo-local"

# ============================================================
# ================ 执行 =======================================
# ============================================================
if __name__ == "__main__":
    if SELECTED_ASR not in AVAILABLE_ASR:
        print(f"错误：未知 ASR 引擎 '{SELECTED_ASR}'，可选: {', '.join(AVAILABLE_ASR)}")
        exit(1)

    asr_cls, asr_kwargs = AVAILABLE_ASR[SELECTED_ASR]
    asr_engine = asr_cls(**asr_kwargs)

    pipeline = BatchTranscriptionPipeline(asr_engine=asr_engine)
    pipeline.run(folder_path=folder_path)
