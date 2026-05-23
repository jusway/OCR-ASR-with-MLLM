"""
一键生成校对稿 — 精准末尾定位版
流程：ASR后1000字定位末尾 → Python截断得精准原文 → 最终校对 → 提取正文 → 元数据
每步输入干净，防止模型在长文本中迷失。
"""

from pathlib import Path

from config.models import AVAILABLE_MODELS
from core.precise_locate_pipeline import PreciseLocatePipeline

# ============================================================
# ================ 以下为全部配置，按需修改 ===================
# ============================================================


# ASR 引擎模型名
asr_model_name = "qwen3-asr-flash-filetrans"



# ============================================================
# ── 步骤0.5：初次校对稿 ───
# ============================================================
DRAFT_SELECTED_MODEL = "deepseek-pro-nothink-stable"
# 输入就已经接近16K，输出再整个十几K，整个上下文接近30K，必须开思考
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
输出格式要按语义自然分段，所有引经据典都用**包裹(md格式)。
直接输出结果，不说任何多余的话。
"""

# ============================================================
# ── 步骤①：末尾定位 ───
# ============================================================
# 讲稿末尾假如说1K，模糊原文参考假如说8K，输出没多少，加起来10K左右。实践后不开思考有概率出错，最好开个思考。
# 开了deepseek-pro-thinking-high还是有概率定错。
LOCATE_SELECTED_MODEL = "deepseek-pro-nothink-stable"
locate_system_prompt = """你是一位严谨的文献整理员。"""
locate_user_prompt_template = """\
【讲稿末尾】
{tail_text}

【模糊原文参考】
{fuzzy_reference_text}

## 背景
讲稿末尾是一位师父在讲解《摩诃止观辅行传弘决辑注》某一集录音的ASR转录稿件，截取了此稿件最后的约{last_n_chars}字。
模糊原文参考是本次讲稿所对应的《摩诃止观辅行传弘决辑注》的大致范围，开头就是上节课结束的地方，为了保证余量，结尾往后延伸，以便完全盖住本集录音。

## 任务
讲稿末尾必定是在讲解模糊原文参考的中间某一片段，找到这个片段，然后原封不动地输出这个片段的结尾一句话即可（原封不动的格式、原封不动的内容、甚至原封不动的注脚）。
（只需要这个片段的结尾一句话，原因是我只需要这一句话复制粘贴搜索定位）
请直接输出结果，不说任何多余的话。
"""
# ============================================================
# ── 步骤②：最终校对稿 ───
# ============================================================
# 录音稿接近16K，原文参考最多5K，输出怎么也得十几K，整个上下文35K吧，必须开始思考。
# 而且实践后 deepseek-high有大概率遗漏一些信息。   #开deepseek-Max拉到极限还是有概率遗漏，可能对于这样30多K上下文遗漏是必然的。
# 试了试kimi-k2.6-nothink,结果里面全是“**次位**”，感觉这模型好傻.      # kimi-k2.5-thinking 也有概率拉胯,输出40%多第二次100%多
FINAL_SELECTED_MODEL = "deepseek-pro-nothink-stable"
final_system_prompt = """\
你是一位极为严谨的文字校对专家。你的唯一职责是将ASR语音识别稿件转写为流畅的书面语。
【核心纪律】
1. 逐字逐句转化：必须保留讲演者的**所有**信息细节、举例和逻辑推演，绝对禁止任何形式的浓缩、总结、概括或删减！
2. 匹配原文：遇到引经据典，请结合提供的【原文参考】进行准确修正，并将引用的经文使用**包裹（例如：**经文内容**）。
3. 净化文本：仅修正错别字、同音字错误，去除口语化的冗余语气词（如“啊”、“那个”、“就是说”等），理顺语序。
你的输出长度必须与输入的ASR片段长度接近，你是一个无损的“文本净化器”，不是“摘要生成器”。"""

final_user_prompt_template = """\
【原文参考】（供全局校对使用）
{precision_text}

【ASR语音识别稿件】
{transcript_text}

## 任务
请对上述【ASR语音识别稿件】进行书面语校对。严格遵守系统提示词中的“绝对禁止删减”原则。
直接输出校对后的正文，按语义自然分段，不要任何客套话、解释或开头结尾的过渡语。
"""

# ============================================================
# ── 步骤③：遗漏检查 ───
# ============================================================
# 逐字稿15K的话，校对稿也十几K，输出基本没多少可以忽略,加起来接近30K,必须开始思考
CHECK_SELECTED_MODEL = "deepseek-pro-nothink-stable"
check_system_prompt = """\
你是一个严格的文稿审查员。你的任务是对比原始的ASR逐字稿和处理后的校对稿。
校对稿的目的是：在保留所有核心法义、比喻、案例的前提下，将口语转化为书面语。
请仔细检查，列出所有在ASR逐字稿中存在、但被校对稿完全删减或遗漏的以下三类信息：
1. 关键的佛学名词/法相名词及其解释。
2. 讲经者为了辅助理解而使用的比喻或通俗案例。
3. 讲经者强调的某些特定逻辑转折。
输出要求：
- 如果没有发现明显遗漏，直接输出"【无遗漏信息】"。
- 如果发现遗漏，请以列表形式输出，要有语音搞和校对稿的对照，方便看出区别。
"""
check_user_prompt_template = """\
【ASR逐字稿】：
{transcript_text}

================

【校对稿】：
{written_text}
"""



# 任务文件夹（脚本自动找 .mp3）
folder_path = Path(r"摩诃止观-定智法师 2017/07正说分/2018年10月29日摩诃止观07正说分第四摄法010")

# 元数据（写入校对稿头部）
year = "2018"
author = "定智法师"
# 截取ASR稿件末尾多少字用于定位末尾
LAST_N_CHARS = 1000

# 遗漏检查开关
ENABLE_CHECK = True  # 设为 False 跳过

# 调试模式（开则每次调大模型前需要确认）
DEBUG_MODE = False
# ============================================================
# ── 执行 ──────────────────
# ============================================================

if __name__ == "__main__":
    pipeline = PreciseLocatePipeline()

    pipeline.run(
        folder_path=folder_path,
        year=year,
        author=author,
        asr_model_name=asr_model_name,
        available_models=AVAILABLE_MODELS,
        locate_model_key=LOCATE_SELECTED_MODEL,
        final_model_key=FINAL_SELECTED_MODEL,
        draft_model_key=DRAFT_SELECTED_MODEL,
        draft_sys_prompt=draft_system_prompt,
        draft_user_prompt_template=draft_user_prompt_template,
        locate_sys_prompt=locate_system_prompt,
        locate_user_prompt_template=locate_user_prompt_template,
        final_sys_prompt=final_system_prompt,
        final_user_prompt_template=final_user_prompt_template,
        check_model_key=CHECK_SELECTED_MODEL if ENABLE_CHECK else None,
        check_sys_prompt=check_system_prompt,
        check_user_prompt_template=check_user_prompt_template,
        last_n_chars=LAST_N_CHARS,
        debug=DEBUG_MODE,
    )
