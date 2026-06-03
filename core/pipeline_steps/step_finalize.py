"""
[8/8] 插入元数据 + 标注原文引用 + 打印汇总。
"""

import time
from core.utils import Color, format_duration
from core.text_marker import mark_quotes


def step_finalize(pipeline, s, final_text, precise_body, transcript_text,
                   year, author, duration, enable_mark_quotes,
                   verify_ok, verify_msg, no_loss, check_text):
    fn = s["files"]["final"]

    # 标注摩诃止观原文引用
    if enable_mark_quotes:
        marked = mark_quotes(final_text, precise_body)
        if marked != final_text:
            count = marked.count('**') // 2
            print(f"{Color.GREEN}✅ 已在最终校对稿中标注 {count} 处摩诃止观原文引用{Color.END}")
            fn.write_text(marked, encoding="utf-8")
            final_text = marked

    # 插入元数据
    print(f"\n[8/8] 插入元数据...")
    current = fn.read_text("utf-8")
    if current.strip().startswith("> 标题："):
        parts = current.split("\n---\n\n", 1)
        current = parts[1] if len(parts) > 1 else current
        print("♻️ 检测到已有旧元数据，替换为本次元数据...")

    body_line = ""
    if precise_body:
        body_line = f"> 摩诃止观正文：{precise_body.replace(chr(10), '')}\n"
    header = (
        f"> 标题：{s['base_name']}\n"
        f"> 时间：{year}\n"
        f"> 时长：{duration}\n"
        f"> 作者：{author}\n"
        f"{body_line}"
        "\n---\n\n"
    )
    fn.write_text(header + current, encoding="utf-8")
    print(f"{Color.GREEN}✅ 元数据已成功插入校对稿开头！{Color.END}")
    if precise_body:
        print("📄 以下为插入的摩诃止观正文：")
        print(body_line.replace("> 摩诃止观正文：", "").strip())

    # ── 汇总 ──
    _print_summary(s, final_text, transcript_text,
                   verify_ok, verify_msg, no_loss, check_text)


def _print_summary(s, final_text, transcript_text,
                    verify_ok, verify_msg, no_loss, check_text):
    print(f"\n{Color.GREEN}===========================================")
    print(f"🎉 全部流程处理完成！{Color.END}")

    clean = final_text
    if "---" in clean:
        clean = clean.split("---", 1)[1]
    clean = clean.replace("**", "").replace("\n", "")
    ratio = len(clean) / len(transcript_text) * 100
    print(f"{Color.ORANGE}📊 字数统计：最终校对稿 {len(clean)} / 逐字稿 {len(transcript_text)} = {ratio:.1f}%{Color.END}")

    print(f"{Color.ORANGE}───────────────────────────────────{Color.END}")
    for label, t in s["step_times"]:
        print(f"{Color.ORANGE}⏱️ {label}: {format_duration(t)}{Color.END}")
    print(f"{Color.ORANGE}⏱️ 全部模型调用总耗时: {format_duration(s['total_time'])}{Color.END}")
    elapsed_real = time.time() - s.get("start_time", 0)
    print(f"{Color.ORANGE}⏱️ 实际运行总耗时: {format_duration(elapsed_real)}{Color.END}")
    print(f"{Color.ORANGE}───────────────────────────────────{Color.END}")

    if verify_ok is not None:
        if verify_ok:
            print(f"{Color.GREEN}  ✓ 编号验证：{verify_msg}{Color.END}")
        else:
            print(f"  ⚠️ 编号验证：{verify_msg}")

    if check_text is not None:
        if no_loss is True:
            print(f"{Color.GREEN}  ✓ 遗漏检查：无关键信息遗漏{Color.END}")
        elif no_loss is False:
            print(f"{Color.ORANGE}  ⚠️ 遗漏检查：可能有遗漏，详见遗漏检查文件{Color.END}")

    print("===========================================\n")
