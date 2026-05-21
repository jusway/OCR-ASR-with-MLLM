"""
一键生成校对稿 — 稳扎版
三步流水线：初次校对 → 精准原文 → 最终校对
每步单一职责，输入干净，输出稳定。
"""

from pathlib import Path

from config.models import AVAILABLE_MODELS
from core.stable_proofread_pipeline import TwoPassProofreadPipeline

# ============================================================
# ================ 以下为全部配置，按需修改 ===================
# ============================================================

# 1) 任务文件夹（脚本自动找 .mp3）
folder_path = Path(r"摩诃止观-定智法师 2017/05正说分第二释名/2018年5月14日摩诃止观05正说分第二释名003")

# 2) 元数据（写入校对稿头部）
year = "2018"
author = "定智法师"

# 3) ASR 引擎模型名
asr_model_name = "qwen3-asr-flash-filetrans"

# 4) 步骤模型配置（模型定义在 config/models.py）
DRAFT_SELECTED_MODEL = "flash-nothink"      # ① 初次校对
# 2 定位精准原文
#
REFINE_SELECTED_MODEL = "flash-nothink"
FINAL_SELECTED_MODEL = "pro-nothink"       # ③ 最终校对

# 调试模式（开则每次调大模型前需要确认）
DEBUG_MODE = True

# 优秀案例
example_path = Path("docs/优秀案例.md")
example_text = example_path.read_text("utf-8").strip() if example_path.exists() else ""

refine_example_path = Path("docs/精准原文案例.md")
refine_example_text = refine_example_path.read_text("utf-8").strip() if refine_example_path.exists() else ""

# ============================================================
# ── 步骤 ①：初次校对稿 ───
# ============================================================

draft_system_prompt = """\
你是一位编辑。你擅长将录音稿件整理成书面语的校对稿，而且不会丢失任何的信息。\
这一点一直是你的骄傲。"""
draft_user_prompt_template = """\
【录音稿件】
{transcript_text}
## 背景
录音稿件是一位师父在讲解《摩诃止观辅行传弘决辑注》的记录。
录音稿件是ASR转录的，包含错别字、同音字和语气词等。
## 任务
将录音稿件整理成书面语的校对稿。
直接输出结果，不说任何多余的话。
"""

# ============================================================
# ── 步骤 ②：精准原文 ───
# ============================================================

refine_system_prompt = """\
你是一位严谨的文献整理员。你擅长从杂乱的原文参考中，精准定位讲稿所对应的那部分原文。"""
refine_user_prompt_template = """\
【参考案例】
{refine_example_text}

【讲稿】
{draft_text}
【模糊原文参考】
{fuzzy_reference_text}
## 背景
讲稿是师父在讲解某一段《摩诃止观辅行传弘决辑注》。
模糊原文参考是我估摸着本次讲解《摩诃止观辅行传弘决辑注》的大致范围，从docx复制到md文件中所得。
模糊原文参考一般包含四部分：摩诃止观正文（一般是被**包裹）、摩诃止观科判（一般在摩诃止观正文前）、摩诃止观解释（一般在摩诃止观正文后）、注脚（一般是罗列在后）。
## 任务
从模糊原文参考中精准截取本讲稿所讲解的范围输出。
输出案例和格式参考"参考案例"。
直接输出结果，不说任何多余的话。
"""

# ============================================================
# ── 步骤 ③：最终校对稿 ───
# ============================================================

final_system_prompt = """\
你是一位编辑。你擅长整理ASR录音稿。"""
final_user_prompt_template = """\
【优秀案例】
{example_text}

【原文参考】
{precision_text}
【ASR录音稿】
{transcript_text}
## 背景
ASR录音稿是一位师父在讲解《摩诃止观辅行传弘决辑注》的记录。
ASR录音稿是ASR转录的，包含错别字、同音字和语气词。
原文参考是师父本次所讲的《摩诃止观辅行传弘决辑注》的原文范围，供你进行参考。
## 任务
参考优秀案例的风格，根据原文参考校对ASR录音稿。
输出完整的校对稿正文。
"""

# ============================================================
# ── 执行 ──────────────────
# ============================================================

if __name__ == "__main__":
    pipeline = TwoPassProofreadPipeline()

    pipeline.run(
        folder_path=folder_path,
        year=year,
        author=author,
        asr_model_name=asr_model_name,
        available_models=AVAILABLE_MODELS,
        draft_model_key=DRAFT_SELECTED_MODEL,
        refine_model_key=REFINE_SELECTED_MODEL,
        final_model_key=FINAL_SELECTED_MODEL,
        draft_sys_prompt=draft_system_prompt,
        draft_user_prompt_template=draft_user_prompt_template,
        refine_sys_prompt=refine_system_prompt,
        refine_user_prompt_template=refine_user_prompt_template,
        final_sys_prompt=final_system_prompt,
        final_user_prompt_template=final_user_prompt_template,
        debug=DEBUG_MODE,
        refine_example_text=refine_example_text,
        example_text=example_text,
    )
