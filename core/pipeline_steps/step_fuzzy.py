"""
[1/8] 模糊原文步骤 —— 等待用户粘贴。
"""

import time
from core.utils import Color, ffprobe_duration


def step_fuzzy_text(pipeline, s):
    fn = s["files"]["fuzzy"]
    print(f"\n[1/8] 粘贴模糊原文到 {fn.name}...")
    if not fn.exists():
        fn.write_text("", encoding="utf-8")
        print(f"{Color.GREEN}✅ 已创建 {fn}{Color.END}")
    else:
        print(f"检测到已存在 {fn.name}")

    duration = ffprobe_duration(s["audio_path"])
    print(f"⏱️ 音频时长: {duration}，供参考选取原文范围")
    print(f"请在 {fn.name} 中整理模糊原文（开头已有上节课结尾，只需粘贴即可），保存后回到此处按回车继续...")
    input()
    s["start_time"] = time.time()
    text = fn.read_text("utf-8").strip()
    if not text:
        print(f"{Color.RED}错误：模糊原文为空，请填入内容后重新运行。")
        exit(1)
    return text, duration
