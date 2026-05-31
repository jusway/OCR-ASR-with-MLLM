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
## 背景
录音稿件是一位师父在讲解《摩诃止观辅行传弘决辑注》的记录。
录音稿件是ASR转录的，包含错别字、同音字和语气词等。

## 角色与任务
你是一位编辑。你擅长将录音稿件整理成书面语的校对稿，而且不会丢失任何的信息。
这一点一直是你的骄傲。
将录音稿件整理成书面语的校对稿。

## 规则
输出格式要按语义自然分段，所有引经据典都用**包裹(md格式)。
直接输出结果，不说任何多余的话。"""
draft_user_prompt_template = """\
【录音稿件】
{transcript_text}
"""

# ============================================================
# ── 步骤①：末尾定位 ───
# ============================================================
# 讲稿末尾假如说1K，模糊原文参考假如说8K，输出没多少，加起来10K左右。   实践后deepseek-pro-nothink有概率出错，最好开个思考。
# 开了deepseek-pro-thinking-high还是有概率错。   deepseek-pro-nothink-stable实践后发现如果模糊原文参考没有片段，也不能变通输出【无匹配片段】，也容易有错
# LOCATE_SELECTED_MODEL = "deepseek-pro-thinking-max"
LOCATE_SELECTED_MODEL = "packy-codex-gpt-5.5-nothink"
locate_system_prompt = """\
## 背景
讲稿末尾是一位师父在讲解《摩诃止观辅行传弘决辑注》某一集录音的ASR转录稿件。
模糊原文参考是本次讲稿所对应的《摩诃止观辅行传弘决辑注》的大致范围，开头就是上节课结束的地方，为了保证余量，结尾往后延伸，以便完全盖住本集录音。
模糊原文参考有四个结构：科判、摩诃止观原文（一般被**包裹）、解释、注脚。

## 角色与任务
你是一位严谨的文献整理员。
讲稿末尾必定是在讲解模糊原文参考的中间某一片段，找到这个片段，然后原封不动地输出这个片段的下一个科判即可。（比如 "△二明大九，初释名略示观法"）
如果没有发现匹配到模糊原文参考的片段，直接输出"【无匹配片段】"。
请直接输出结果，不说任何多余的话。"""
locate_user_prompt_template = """\
【讲稿末尾（截取最后{last_n_chars}字）】
{tail_text}

【模糊原文参考】
{fuzzy_reference_text}
"""
# ============================================================
# ── 步骤②：最终校对稿 ───
# ============================================================
# 录音稿接近16K，原文参考最多5K，输出怎么也得十几K，整个上下文35K吧，必须开始思考。
# 而且实践后 deepseek-high有大概率遗漏一些信息。   #开deepseek-Max拉到极限还是有概率遗漏，可能对于这样30多K上下文遗漏是必然的。
# 试了试kimi-k2.6-nothink,结果里面全是“**次位**”，感觉这模型好傻.      # kimi-k2.5-thinking 也有概率拉胯,输出40%多第二次100%多
# FINAL_SELECTED_MODEL = "deepseek-pro-nothink-stable" # 浓缩率经常20%，太尴尬了
# FINAL_SELECTED_MODEL = "packy-codex-gpt-5.5-nothink" # 偶尔会浓缩率20%多，尴尬，太过浓缩
# FINAL_SELECTED_MODEL = "packy-codex-gpt-5.4-high" # 第一次用浓缩率40%，尴尬
FINAL_SELECTED_MODEL = "deepseek-pro-nothink-stable"
final_system_prompt = """\
## 背景
【ASR语音识别稿件】是一位师父在讲解《摩诃止观辅行传弘决辑注》的记录。
【原文参考】是本节课的《摩诃止观辅行传弘决辑注》的节选，有四个结构：
科判（类似“△二衣食具足二，初来意”）、摩诃止观原文（比如“**得此三谛三昧，故名王三昧，一切三昧悉入其中。**”）、解释、注脚。

## 任务
请逐字逐句对【ASR语音识别稿件】进行校对：修正错别字、同音字错误，去除多余啰嗦的语气词。
保留ASR语音识别稿件的**所有信息**，包括但不限于：观点、通俗案例、类比比喻、\
解释、即兴讲话、开玩笑的话、开经偈和回向偈，等等。
虽然语音稿没有这么做，但是你要把所有摩诃止观原文作为小节标题，使用双星号包裹，插入到合适的位置。
直接输出校对后的正文，按语义自然分段。"""

final_user_prompt_template = """\
【原文参考】（供全局校对使用）
{precision_text}

【ASR语音识别稿件】
{transcript_text}
"""

# ============================================================
# ── 步骤③：遗漏检查 ───
# ============================================================
# 逐字稿15K的话，校对稿也十几K，输出基本没多少可以忽略,加起来接近30K,必须开始思考
CHECK_SELECTED_MODEL = "packy-codex-gpt-5.5-nothink"
check_system_prompt = """\
你是一个严格的文稿审查员。你的任务是对比原始的ASR逐字稿和处理后的校对稿。
校对稿的目的是：保留所有法义、比喻、案例、解释、即兴讲话等等，将口语转化为书面语。
请仔细检查，列出所有在ASR逐字稿中存在、但被校对稿完全删减或遗漏的信息。
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
folder_path = Path(r"../摩诃止观-定智法师 2017/11正说分/2019年3月15日摩诃止观正说分11第七正修阴入界境观不思议境005")

# 元数据（写入校对稿头部）
year = "2019"
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
