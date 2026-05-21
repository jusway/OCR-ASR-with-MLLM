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
# 1 得到最初校对稿。还是要使用PRO , flash模型转出来的，有错字不改正，可能模型觉得那是对的
DRAFT_SELECTED_MODEL = "pro-nothink"
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
# 必须要用PRO，要不然用flash连智商不够，连你的命令都听不懂
REFINE_SELECTED_MODEL = "pro-nothink"
refine_system_prompt = """\
你是一位严谨的文献整理员。"""
refine_user_prompt_template = """\
【讲稿】
{draft_text}
【模糊原文参考】
{fuzzy_reference_text}
## 背景
讲稿是师父在讲解某一段《摩诃止观辅行传弘决辑注》（可能存在错别字等谬误）。
模糊原文参考是我估摸着本次讲解《摩诃止观辅行传弘决辑注》的大致范围，从docx复制到md文件中所得。
模糊原文参考一般包含四部分：摩诃止观正文（一般是被**包裹）、摩诃止观科判（一般在摩诃止观正文前）、摩诃止观解释（一般在摩诃止观正文后）、注脚（一般是罗列在后）。
## 任务
从模糊原文参考中，精准截取本讲稿所讲解的范围，而把没有讲到的部分砍掉，按照下面的**输出格式案例**进行输出。
尤其注意把每一个注脚放到对应有标注的段落的下方。
直接输出结果，不说任何多余的话。
## 输出格式案例
△二各释三，初贯穿
**贯穿义者，智慧利用，穿灭烦恼。《大经》云：“利镢斲地，磐石砂砾，直至金刚。”《法华》云：“穿凿高原，犹见干燥土，施功不已，遂渐至泥。”此就所破得名，立贯穿观也。**
初观中云利镢等者，大鉏曰镢[[1\]](#_ftn1)。此琢，治玉耳，非今意也，应作此斲。斲者，破也。大石曰磐，粉石曰砂，小石曰砾。《大经》十八《性品》云[[2\]](#_ftn2)：“譬如有人，善知伏藏，即以利镢斲地，磐石砂砾，直过无难，唯至金刚不能穿彻。”今借彼譬，总含三惑[[3\]](#_ftn3)，准止可知。应置金刚，但存穿彻，即此中意。
[[1\]](#_ftnref1) 大鉏曰镢：鉏chú：锄具也。镢：音jué。斲：同“斫zhuó”。
[[2\]](#_ftnref2) 《大经》十八《性品》云：今《大正藏》位于《大经》卷八·《如来性品第十二》。金刚不能彻者，经中以喻佛性故也。
[[3\]](#_ftnref3) 今借彼譬，总含三惑：《维摩文疏》二十三：“《大涅槃》云：‘即用利镢斵之，磐石砂砾，直过无碍，彻至金刚。’菩萨从假入空观时，贯穿俗谛见思之磐石，滞真无知之砂，无明覆蔽一实谛之砾，彻洞无碍，即是穷至心性本际金刚也。”
次引《法华》穿凿义边，以证贯穿。未分干湿，故亦通也。
△二观逹
**观达义者，观智通达，契会真如。《瑞应》云：“息心达本源，故号为沙门。”《大论》云：“清净心常一，则能见般若。”此就能观得名，故立观达观也。**
次释观达中引《瑞应》文，言虽有息，意存达边。
"""

# ============================================================
# ── 步骤 ③：最终校对稿 ───
# ============================================================
FINAL_SELECTED_MODEL = "pro-nothink"       # ③ 最终校对
final_system_prompt = """\
你是一位编辑。你擅长整理ASR录音稿，得到书面校对稿。"""
final_user_prompt_template = """\
【原文参考】
{precision_text}
【ASR录音稿】
{transcript_text}
## 背景
ASR录音稿是一位师父在讲解《摩诃止观辅行传弘决辑注》的记录。
ASR录音稿是ASR转录的，包含错别字、同音字和语气词。
原文参考是师父本次所讲的《摩诃止观辅行传弘决辑注》的原文范围，供你进行参考。
## 任务
根据原文参考，校对ASR录音稿，输出校对稿。
直接输出结果，不说任何多余的话。
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
