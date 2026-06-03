"""
[0/8] ASR 转录步骤（带缓存）。
"""

import time
from core.utils import Color, format_duration


def step_asr(pipeline, s):
    fn = s["files"]["transcript"]
    print(f"\n[0/8] ASR 转录 → 逐字稿.md...")
    if fn.exists():
        print(f"{Color.GREEN}✅ 检测到已存在逐字稿，直接复用: {fn.name}{Color.END}")
        return fn.read_text("utf-8")

    print("正在调用 ASR 引擎生成逐字稿...")
    t0 = time.time()
    text = s["asr_engine"].recognize(s["audio_path"])
    elapsed = time.time() - t0
    s["total_time"] += elapsed
    s["step_times"].append(("[0/8] ASR 转录", elapsed))
    print(f"⏱️ ASR 转录耗时: {format_duration(elapsed)}")
    fn.write_text(text, encoding="utf-8")
    print(f"{Color.GREEN}逐字稿已保存至: {fn}{Color.END}")
    return text
