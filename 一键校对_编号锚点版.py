"""
一键校对 — 编号锚点版（完整流程）
将 ASR 稿件按句编号后全量送入 LLM，要求输出 1:1 保留编号，
脚本自动验证编号完整性 + 最终去编号，用硬约束防止长文本信息遗漏。

完整流程：
  [0/8] ASR 转录
  [1/8] 模糊原文
  [2/8] 脚注重排
  [3/8] 末尾定位
  [4/8] Python 截断得精准原文
  [5/8] 编号锚点校对
  [6/8] 遗漏检查
  [7/8] 脚本去编号
  [8/8] 插入元数据
"""

from pathlib import Path

from config.models import AVAILABLE_MODELS
from core.anchor_proofread_pipeline import AnchorProofreadPipeline

# ============================================================
# ================ 以下为全部配置，按需修改 ===================
# ============================================================

# 1) 任务文件夹（脚本自动找 .mp3）
folder_path = Path(r"摩诃止观-定智法师 2017/09正说分/2019年1月1日摩诃止观09正说分第六方便044")

# 2) 元数据
year = "2019"
author = "定智法师"

# 3) ASR 引擎模型名
asr_model_name = "qwen3-asr-flash-filetrans"

# 4) 截取 ASR 稿件末尾多少字用于定位末尾
LAST_N_CHARS = 1000

# 5) 调试模式
DEBUG_MODE = False


# ============================================================
# ── 步骤 1：末尾定位 ───
# ============================================================
LOCATE_SELECTED_MODEL = "deepseek-pro-thinking-max"
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
# ── 步骤 2：编号锚点校对（★核心变更）───
# ============================================================
ANCHOR_SELECTED_MODEL = "deepseek-pro-thinking-high"
anchor_system_prompt = """\
## 背景
【ASR语音识别稿件】是一位师父在讲解《摩诃止观辅行传弘决辑注》的记录，每一句话都有对应的编号。
【原文参考】是本节课的《摩诃止观辅行传弘决辑注》的节选，作用是本次任务的参考，有四个结构：
科判（类似"△二衣食具足二，初来意"）、摩诃止观原文（比如"**得此三谛三昧，故名王三昧，一切三昧悉入其中。**"）、解释、注脚。

## 角色与任务
- 你是严谨的文字编辑。请将带有编号的ASR语音识别稿件逐句校对，输出为正式的书面校对稿。
- 校对错别字和同音字、去掉过于啰嗦的语气词、想办法使句子通顺易读。
- 保留ASR语音识别稿件的**所有信息**，包括但不限于：观点、通俗案例、类比比喻、\
解释、即兴讲话、开玩笑的话、开经偈和回向偈，等等。
- **编号对齐**：语音稿已逐句编号。输出的校对稿，必须保留每一个编号，一个编号也不能少！
- 虽然一个编号也不能少，但是当认为某编号句子应该删节的时候，允许在编号后面输出空字符来代表删节。\
允许在不同编号之间交换句子。这些变通都是为了书面稿的通顺易读，但绝不能丢失信息！

## 输出格式
格式参考语音搞，严格按照编号顺序逐句输出。直接输出结果，不说任何多余的话。"""

anchor_user_prompt_template = """\
【原文参考】（供全局校对使用）
{precision_text}

【ASR语音识别稿件】
{transcript_text}
"""


# ============================================================
# ── 步骤 3：遗漏检查 ───
# ============================================================
CHECK_SELECTED_MODEL = "deepseek-pro-thinking-high"
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


# ============================================================
# ── 执行 ──────────────────
# ============================================================

if __name__ == "__main__":
    pipeline = AnchorProofreadPipeline()

    pipeline.run(
        folder_path=folder_path,
        year=year,
        author=author,
        asr_model_name=asr_model_name,
        available_models=AVAILABLE_MODELS,

        locate_model_key=LOCATE_SELECTED_MODEL,
        locate_sys_prompt=locate_system_prompt,
        locate_user_prompt_template=locate_user_prompt_template,
        anchor_model_key=ANCHOR_SELECTED_MODEL,
        anchor_sys_prompt=anchor_system_prompt,
        anchor_user_prompt_template=anchor_user_prompt_template,
        check_model_key=CHECK_SELECTED_MODEL,
        check_sys_prompt=check_system_prompt,
        check_user_prompt_template=check_user_prompt_template,
        last_n_chars=LAST_N_CHARS,
        debug=DEBUG_MODE,
    )
