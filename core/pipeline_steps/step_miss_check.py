"""
[6/8] 遗漏检查步骤（带缓存）。
"""

from core.utils import Color


def step_miss_check(pipeline, s, transcript_text, raw_output,
                     check_sys_prompt, check_user_prompt_template):
    fn = s["files"]["check"]
    print(f"\n[6/8] 遗漏检查...")
    engine = pipeline._init_model(s, "遗漏检查", s["check_model_key"])
    if not (engine and check_sys_prompt and check_user_prompt_template):
        return None, None

    if fn.exists():
        report = fn.read_text("utf-8").strip()
        print(f"{Color.GREEN}✅ 检测到已存在遗漏检查报告，直接复用: {fn.name}{Color.END}")
        return None, report  # cached, no_loss unknown

    pipeline._confirm_step("遗漏检查", s["debug"])
    prompt = check_user_prompt_template.format(
        transcript_text=transcript_text,
        written_text=raw_output,
    )
    report, elapsed = pipeline._llm_generate(
        s, engine, check_sys_prompt, prompt,
        "[6/8] 遗漏检查", "第6步_", "遗漏检查",
    )
    fn.write_text(report, encoding="utf-8")
    print(f"{Color.GREEN}✅ 遗漏检查报告已保存至: {fn.name}{Color.END}")

    if "【无遗漏信息】" in report or "完美" in report:
        print(f"{Color.GREEN}  ✓ 检查结果：无关键信息遗漏{Color.END}")
        return True, report.strip()
    print("  ⚠️ 检查结果：可能有遗漏，请查看报告")
    return False, report.strip()
