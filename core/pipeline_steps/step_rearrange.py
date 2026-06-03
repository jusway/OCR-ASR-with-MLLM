"""
[2/8] 脚注重排步骤（带缓存）。
"""

from core.utils import Color
from 常用脚本.脚注重排 import rearrange_footnotes


def step_rearrange(pipeline, s, fuzzy_ref):
    fn = s["parent_dir"] / "第2步_模糊原文_已重排.md"
    print(f"\n[2/8] 脚注重排 → {fn.name}...")
    if fn.exists():
        print(f"{Color.GREEN}✅ 检测到已存在重排后模糊原文，直接复用: {fn.name}{Color.END}")
        return fn.read_text("utf-8").strip()

    result = rearrange_footnotes(fuzzy_ref)
    fn.write_text(result, encoding="utf-8")
    print(f"{Color.GREEN}✅ 脚注重排完成 → {fn.name}{Color.END}")
    return result
