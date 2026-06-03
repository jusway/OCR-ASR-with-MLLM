"""
[7/8] 脚本去编号步骤（带缓存）。
"""

import re
from core.utils import Color


def step_remove_numbers(pipeline, s, raw_output):
    fn = s["files"]["final"]
    print(f"\n[7/8] 脚本去除编号 → 最终校对稿.md...")
    if fn.exists():
        print(f"{Color.GREEN}✅ 检测到已存在最终校对稿，直接复用: {fn.name}{Color.END}")
        return fn.read_text("utf-8")

    parts = []
    for m in re.finditer(r'\[(\d{4})\]\s*', raw_output):
        start = m.end()
        next_match = re.search(r'\[(\d{4})\]\s*', raw_output[start:])
        end = start + next_match.start() if next_match else len(raw_output)
        content = raw_output[start:end].strip()
        if content:
            parts.append(content)

    merged = pipeline._merge_sentences(parts)
    final_text = "\n\n".join(merged) if merged else raw_output
    fn.write_text(final_text, encoding="utf-8")
    print(f"{Color.GREEN}✅ 最终校对稿已保存至: {fn.name}{Color.END}")
    return final_text
