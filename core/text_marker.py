"""
文本标注工具：在目标文本中标注与原文匹配的片段。
用于在最终校对稿中用 ** 包裹摩诃止观原文引用。
"""

import re


def mark_quotes(final_text: str, precise_body: str) -> str:
    """用 ** 标注 final_text 中与 precise_body 匹配的原文片段。"""
    if not precise_body:
        return final_text

    # 细粒度切分 precise_body（按任意标点切分，短片段匹配率更高）
    segments = re.split(r'(?<=[。！？，、；：])', precise_body)
    segments = [s.strip() for s in segments if s.strip() and len(s.strip()) >= 4]

    # 收集匹配位置
    markers = []
    for seg in segments:
        idx = final_text.find(seg)
        if idx != -1:
            markers.append((idx, idx + len(seg)))
            continue
        # 最长公共子串扫描
        min_len = max(int(len(seg) * 0.5), 5)
        best = None
        for w in range(len(seg), min_len - 1, -1):
            for i in range(len(seg) - w + 1):
                sub = seg[i:i + w]
                idx = final_text.find(sub)
                if idx != -1:
                    best = (idx, idx + w)
                    break
            if best:
                break
        if best:
            markers.append(best)

    if not markers:
        return final_text

    # 合并重叠标记
    markers.sort()
    merged = []
    for s, e in markers:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    # 从后往前插入 **
    chars = list(final_text)
    for s, e in reversed(merged):
        chars.insert(e, '**')
        chars.insert(s, '**')
    return ''.join(chars)
