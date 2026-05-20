from pathlib import Path

from core.pipeline import run
from core.text_engine import DeepSeekText, GeminiText, NvidiaText

# ============================================================
# ================ 以下为全部配置，按需修改 ===================
# ============================================================

# 1) 任务文件夹（脚本自动找 .mp3）
folder_path = Path(r"摩诃止观-定智法师 2017/04正说分第一大意感大果裂大网归大处/2018年5月8日摩诃止观04正说分第一大意感大果裂大网归大处001")

# 2) 元数据（写入校对稿头部）
year = "2018"
author = "定智法师"

# 3) ASR 引擎模型名
asr_model_name = "qwen3-asr-flash-filetrans"

# 4) 文本模型：想换模型只改 SELECTED_MODEL 这一行的 key
#    Pro = deepseek-v4-pro, Flash = deepseek-v4-flash
#    思考模式 effort: max / high；非思考模式无需 reasoning_effort
AVAILABLE_MODELS = {
    # ── Pro ──
    "pro-nothink": (DeepSeekText, {
        "model_name": "deepseek-v4-pro",
        "enable_thinking": False,
    }),
    "pro-high": (DeepSeekText, {
        "model_name": "deepseek-v4-pro",
        "enable_thinking": True, "reasoning_effort": "high",
    }),
    "pro-max": (DeepSeekText, {
        "model_name": "deepseek-v4-pro",
        "enable_thinking": True, "reasoning_effort": "max",
    }),
    # ── Flash ──
    "flash-nothink": (DeepSeekText, {
        "model_name": "deepseek-v4-flash",
        "enable_thinking": False,
    }),
    "flash-high": (DeepSeekText, {
        "model_name": "deepseek-v4-flash",
        "enable_thinking": True, "reasoning_effort": "high",
    }),
    "flash-max": (DeepSeekText, {
        "model_name": "deepseek-v4-flash",
        "enable_thinking": True, "reasoning_effort": "max",
    }),
}
COVERAGE_SELECTED_MODEL = "pro-nothink"
SELECTED_MODEL = "pro-nothink"
VERIFIER_SELECTED_MODEL = "pro-nothink"
EXTRACT_SELECTED_MODEL = "pro-nothink"

# 模糊原文种子（首次运行时自动创建 模糊原文.md，此后以文件内容为准）
reference_text = ""

# 优秀案例（从 优秀案例.md 读取，作为 one-shot 示例拼接在逐字稿前）
example_path = Path("docs/优秀案例.md")
example_text = example_path.read_text("utf-8").strip() if example_path.exists() else ""

# 覆盖度检查：找出录音稿件中原文参考未覆盖的内容
coverage_system_prompt = """\
你是一位严谨的审核员。
你习惯只输出遗漏结论，不说一点多余的话。"""
coverage_user_prompt_template = """\
【录音稿件】
{transcript_text}
【原文参考】
{fuzzy_reference_text}
## 背景
【录音稿件】是一位师父在讲解《摩诃止观辅行传弘决辑注》的记录。
【录音稿件】是ASR转录的，包含大量错别字、同音字和语气词，请根据语义理解而非字面匹配。
【原文参考】是这位师父本次讲解的《摩诃止观辅行传弘决辑注》的大致原文范围，是我复制给你看的。
## 任务
逐段检查【录音稿件】中讲解的内容，找出【原文参考】是不是复制的范围不够广。（讲解已经讲到后面的内容，但是复制范围不够所以没有后面了）
（稿件开头明显存在“前情提要”的部分，那个属于上节课，不在本节课考虑范围）
如果发现复制范围不够广，请你输出遗漏了哪些需要复制的信息。
如果确认录音稿讲讲解没有超出【原文参考】，那就输出“【无遗漏信息】”。
"""

# 录音稿件+大致原文=得到校对稿
text_system_prompt = """\
你是一位编辑。你擅长校对录音稿件，整理得井井有条，而且不会丢失任何的信息。这一点一直是你的骄傲。
"""
text_user_prompt_template = """\
【优秀案例】
{example_text}
【原文参考】
{fuzzy_reference_text}
【录音稿件】
{transcript_text}
## 背景
【录音稿件】是一位师父在讲解《摩诃止观辅行传弘决辑注》的记录。
【原文参考】是这位师父讲解的《摩诃止观辅行传弘决辑注》的大致原文范围，供你参考。
## 任务
参考【优秀案例】，校对【录音稿件】，直接输出校对稿的正文内容。
"""

# 信息丢失检查提示词
verifier_system_prompt = """\
你是一位编辑。你擅长对照录音稿件和校对稿，进行审核工作。
而且你要求非常严格，这一点一直是你的骄傲。"""
verifier_user_prompt_template = """
【录音稿件】
{transcript_text}
【校对稿】
{written_text}
## 背景
录音稿件是一位师父在讲解《摩诃止观辅行传弘决辑注》的记录。
校对稿是对录音稿件进行校对得到的。
## 任务
判断录音稿件的信息，是否被校对稿遗漏了。
如果确认没有遗漏信息，直接输出“【无遗漏信息】”。
如果有遗漏，请详细解释遗漏了什么。
"""

# 精准定位并提取纯原文（一次调用）
extract_system_prompt = """你是一位严谨的文献整理员。"""
extract_user_prompt_template = """\
## 背景
讲稿是一位师父在讲解《摩诃止观辅行传弘决辑注》。
复制的节选是我估摸着这个讲稿所讲的《摩诃止观辅行传弘决辑注》的大致范围，复制给你看。\
是docx文档节选复制到markdown文件中得到的文本，可能会有些乱码。
复制的节选包括四部分，摩诃止观正文（一般被**包裹住了）、摩诃止观科判（一般跟在正文前面）、\
摩诃止观解释（一般跟在正文后面）、注脚（一般在一起罗列）
## 任务
找到【讲稿】的内容对应在讲解【复制的节选】中摩诃止观正文的哪些句子（按顺序）。
(讲稿最后如果存在只是提了一下的正文，但是并没有开讲，这是下节课要讲的，本节课不纳入)
然后把这些摩诃止观正文按顺序直接拼接输出。

【讲稿】
{written_text}

【复制的节选】
{fuzzy_reference_text}
"""

if __name__ == "__main__":
    run(
        folder_path=folder_path,
        year=year,
        author=author,
        reference_text=reference_text,
        asr_model_name=asr_model_name,
        available_models=AVAILABLE_MODELS,
        selected_model=SELECTED_MODEL,
        coverage_selected_model=COVERAGE_SELECTED_MODEL,
        verifier_selected_model=VERIFIER_SELECTED_MODEL,
        extract_selected_model=EXTRACT_SELECTED_MODEL,
        coverage_system_prompt=coverage_system_prompt,
        coverage_user_prompt_template=coverage_user_prompt_template,
        text_system_prompt=text_system_prompt,
        text_user_prompt_template=text_user_prompt_template,
        verifier_system_prompt=verifier_system_prompt,
        verifier_user_prompt_template=verifier_user_prompt_template,
        extract_system_prompt=extract_system_prompt,
        extract_user_prompt_template=extract_user_prompt_template,
        example_text=example_text,
    )
