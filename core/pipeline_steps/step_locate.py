"""
[3/8] 末尾定位 + [4/8] 精准原文截断（带缓存）。
"""

import re
from core.utils import Color


def step_locate_and_precision(pipeline, s, fuzzy_ref, transcript_text,
                               locate_sys_prompt, locate_user_prompt_template,
                               last_n_chars):
    print(f"\n[3/8] 末尾定位 → 末尾定位.md...")
    engine = pipeline._init_model(s, "末尾定位", s["locate_model_key"])
    precision_fn = s["files"]["precision"]

    if precision_fn.exists():
        print(f"{Color.GREEN}✅ 检测到已存在精准原文，直接复用: {precision_fn.name}{Color.END}")
        precision_text = precision_fn.read_text("utf-8")
        precise_body = pipeline._load_or_extract_precise_body(s, precision_text)
        return precision_text, precise_body

    # 末尾定位
    pipeline._confirm_step("末尾定位", s["debug"])
    tail_text = transcript_text[-last_n_chars:]
    prompt = locate_user_prompt_template.format(
        tail_text=tail_text,
        fuzzy_reference_text=fuzzy_ref,
        last_n_chars=last_n_chars,
    )
    result, elapsed = pipeline._llm_generate(
        s, engine, locate_sys_prompt, prompt,
        "[3/8] 末尾定位", "第3步_", "末尾定位",
    )
    last_sentence = result.strip()
    locate_fn = s["parent_dir"] / f"第3步_{s['base_name']}_末尾定位.md"
    locate_fn.write_text(last_sentence, encoding="utf-8")
    print(f"{Color.GREEN}✅ 末尾已保存至: {locate_fn.name}{Color.END}")

    # [4/8] 精准原文截断
    print(f"\n[4/8] Python 截断 → 精准原文.md...")
    if last_sentence == "【无匹配片段】":
        print("⚠️ 模型返回「无匹配片段」，精准原文和正文将为空。")
        precision_text = "暂无参考资料"
        precise_body = ""
    else:
        idx = fuzzy_ref.find(last_sentence)
        if idx == -1:
            print(f"{Color.RED}❌ 未在模糊原文中找到末尾，模型返回的结果不正确。请检查末尾定位结果后重试。{Color.END}")
            exit(1)
        precision_text = fuzzy_ref[:idx + len(last_sentence)]
        precise_body = pipeline._extract_precise_body(precision_text)

    precision_fn.write_text(precision_text, encoding="utf-8")
    print(f"{Color.GREEN}✅ 精准原文已保存至: {precision_fn.name}{Color.END}")

    precise_body_fn = s["files"]["precise_body"]
    precise_body_fn.write_text(precise_body, encoding="utf-8")
    if precise_body:
        print(f"{Color.GREEN}✅ 精准原文正文已提取至: {precise_body_fn.name}（{len(precise_body)} 字）{Color.END}")

    return precision_text, precise_body
