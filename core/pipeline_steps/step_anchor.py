"""
[5/8] 编号锚点校对步骤（核心）。
"""

import re
from core.utils import Color


def step_anchor_proofread(pipeline, s, transcript_text, precision_text,
                           anchor_sys_prompt, anchor_user_prompt_template,
                           min_chars_per_item):
    print(f"\n[5/8] 编号锚点校对...")

    # 5a. 编号稿（带缓存）
    numbered_fn = s["files"]["numbered"]
    if numbered_fn.exists():
        numbered_text = numbered_fn.read_text("utf-8")
        raw_sentences = re.split(r'\n(?=\[\d{4}\])', numbered_text)
        raw_sentences = [x.strip() for x in raw_sentences if x.strip()]
        n_ratio = len(numbered_text) / len(transcript_text) * 100
        print(f"{Color.GREEN}✅ 检测到已存在编号稿，直接复用: {numbered_fn.name}{Color.END}")
        print(f"📊 字数统计：编号稿 {len(numbered_text)} / 逐字稿 {len(transcript_text)} = {n_ratio:.1f}%")
    else:
        numbered_text, raw_sentences = pipeline._build_numbered_draft(
            transcript_text, min_chars_per_item,
        )
        numbered_fn.write_text(numbered_text, encoding="utf-8")
        n_ratio = len(numbered_text) / len(transcript_text) * 100
        avg_chars = sum(len(x) for x in raw_sentences) / len(raw_sentences) if raw_sentences else 0
        print(f"{Color.GREEN}✅ 编号稿已生成: {numbered_fn.name}（共 {len(raw_sentences)} 项，均长 {avg_chars:.0f} 字，阈值 {min_chars_per_item} 字，总字数 {len(numbered_text)} / 逐字稿 {len(transcript_text)} = {n_ratio:.1f}%）{Color.END}")

    total_sentences = len(raw_sentences)

    # 5b. LLM 调用（带缓存 + 浓缩率检查）
    engine = pipeline._init_model(s, "编号锚点校对", s["anchor_model_key"])
    output_fn = s["files"]["anchor_output"]
    raw_output = None

    if output_fn.exists():
        raw_output = output_fn.read_text("utf-8")
        a_ratio = len(raw_output) / len(numbered_text) * 100
        print(f"{Color.GREEN}✅ 检测到已存在编号校对稿，直接复用: {output_fn.name}{Color.END}")
        print(f"📊 浓缩率：编号校对稿 {len(raw_output)} / 编号稿 {len(numbered_text)} = {a_ratio:.1f}%")
        if pipeline._should_regenerate_concentration(a_ratio):
            output_fn.unlink()
            raw_output = None
            print(f"{Color.ORANGE}♻️ 删除旧编号校对稿，重新生成...{Color.END}")

    if raw_output is None:
        raw_output = pipeline._call_anchor_llm(
            s, engine, anchor_sys_prompt, anchor_user_prompt_template,
            numbered_text, total_sentences, precision_text,
            "[5/8] 编号锚点校对",
        )
        a_ratio = len(raw_output) / len(numbered_text) * 100
        print(f"{Color.GREEN}✅ 编号校对稿已保存至: {output_fn.name}{Color.END}")
        print(f"📊 浓缩率：编号校对稿 {len(raw_output)} / 编号稿 {len(numbered_text)} = {a_ratio:.1f}%")
        if pipeline._should_regenerate_concentration(a_ratio):
            print(f"{Color.ORANGE}♻️ 重新生成编号校对稿...{Color.END}")
            raw_output = pipeline._call_anchor_llm(
                s, engine, anchor_sys_prompt, anchor_user_prompt_template,
                numbered_text, total_sentences, precision_text,
                "[5/8] 编号锚点校对(重试)",
            )
            a_ratio = len(raw_output) / len(numbered_text) * 100
            print(f"📊 重新生成后浓缩率：编号校对稿 {len(raw_output)} / 编号稿 {len(numbered_text)} = {a_ratio:.1f}%")

    # 5c. 编号完整性验证
    verify_ok, verify_msg, duplicate = pipeline._verify_numbering(raw_output, total_sentences)

    # 编号重复 → 可选重试
    if duplicate:
        choice = input(
            f"{Color.ORANGE}⚠️ 检测到 {len(duplicate)} 处编号重复，是否重新生成编号校对稿？(y/n): {Color.END}"
        ).strip().lower()
        if choice == "y":
            print(f"{Color.ORANGE}♻️ 重新生成编号校对稿...{Color.END}")
            output_fn.unlink()
            raw_output = pipeline._call_anchor_llm(
                s, engine, anchor_sys_prompt, anchor_user_prompt_template,
                numbered_text, total_sentences, precision_text,
                "[5/8] 编号锚点校对(重复重试)",
            )
            a_ratio = len(raw_output) / len(numbered_text) * 100
            print(f"📊 重新生成后浓缩率：编号校对稿 {len(raw_output)} / 编号稿 {len(numbered_text)} = {a_ratio:.1f}%")
            verify_ok, verify_msg, duplicate = pipeline._verify_numbering(raw_output, total_sentences)

    return raw_output, verify_ok, verify_msg
