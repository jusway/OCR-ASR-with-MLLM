"""
将模糊原文末尾罗列的脚注定义，搬回到正文中对应标注的后方。
输入：Word 复制格式（脚注全部堆在末尾）
输出：已重排（每个脚注紧跟其标注位置）
"""
import re
import sys
from pathlib import Path


def rearrange_footnotes(text: str) -> str:
    """
    1. 把正文和末尾脚注附录切开
    2. 解析附录里每条脚注的正文
    3. 把脚注体插回正文标注之后
    """
    # ── 1. 切分：正文 vs 脚注附录 ──
    split = re.search(r'\n(?=\[\[\d+\\\]\]\(#_ftnref\d+\))', text)
    if not split:
        return text  # 无脚注，原样返回

    body = text[:split.start()]
    appendix = text[split.start():]

    # ── 2. 解析附录：{"1": "脚注体...", "2": "脚注体..."} ──
    fn_defs = {}
    fn_start = re.compile(r'^\[\[(\d+)\\\]\]\(#_ftnref\1\)')
    appendix_lines = appendix.splitlines()
    i = 0
    while i < len(appendix_lines):
        line = appendix_lines[i].strip()
        m = fn_start.match(line)
        if m:
            num = m.group(1)
            body_lines = [appendix_lines[i]]
            j = i + 1
            while j < len(appendix_lines):
                nxt = appendix_lines[j].strip()
                if fn_start.match(nxt):
                    break
                body_lines.append(appendix_lines[j])
                j += 1
            fn_defs[num] = "\n".join(body_lines)
            i = j
        else:
            i += 1

    if not fn_defs:
        return text

    # ── 3. 把 [任意内容](#_ftnN) 替换为 [注脚：脚注正文] ──
    def repl(m):
        num = m.group(1)
        if num in fn_defs:
            fn_body = fn_defs[num]
            # 取脚注定义行之后的内容（跳过 "[[N\]](#_ftnrefN) " 前缀）
            fn_body = re.sub(r'^\[\[\d+\\\]\]\(#_ftnref\d+\)\s*', '', fn_body)
            return f"[注脚：{fn_body}]"
        return m.group(0)

    result = re.sub(r'\[.*?\]\(#_ftn(\d+)\)', repl, body)

    # 清理尾部残留（--- 分隔线、多余空行）
    result = re.sub(r'\n*-{3,}\s*$', '', result)
    return result.strip() + "\n"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python 脚注重排.py <模糊原文.md>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = input_path.parent / input_path.name.replace(".md", "_已重排.md")
    
    text = input_path.read_text("utf-8")
    rearranged = rearrange_footnotes(text)
    output_path.write_text(rearranged, encoding="utf-8")
    print(f"✅ 脚注重排完成 → {output_path.name}")
