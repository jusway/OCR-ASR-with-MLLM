from pathlib import Path

from core.pipeline import run
from core.text_engine import DeepSeekText, GeminiText, NvidiaText

# ============================================================
# ================ 以下为全部配置，按需修改 ===================
# ============================================================

# 1) 任务文件夹（脚本自动找 .mp3）
folder_path = Path(r"摩诃止观-定智法师 2017/03正说分第一大意发大心/摩诃止观03正说分/2018年1月15日摩诃止观03正说分第一大意修大行002")

# 2) 元数据（写入校对稿头部）
year = "2018"
author = "定智法师"

# 3) ASR 引擎模型名
asr_model_name = "qwen3-asr-flash-filetrans"

# 4) 文本模型：想换模型只改 SELECTED_MODEL 这一行的 key
AVAILABLE_MODELS = {
    "deepseek": (DeepSeekText, {
        "model_name": "deepseek-v4-pro",
        "enable_thinking": True, "reasoning_effort": "max",
    }),
    "deepseek-fast": (DeepSeekText, {
        "model_name": "deepseek-v4-flash",
        "enable_thinking": True,"reasoning_effort": "high",
    }),
}
SELECTED_MODEL = "deepseek"
FAST_SELECTED_MODEL = "deepseek"

# 模糊原文种子（首次运行时自动创建 模糊原文.md，此后以文件内容为准）
reference_text = ""

# 逐字稿+大致原文=得到校对稿
text_system_prompt = '你是一位严谨的校对员。'
text_user_prompt_template = """\
## 背景
逐字稿是一位师父在讲解《摩诃止观辅行传弘决辑注》的记录，有很多转录错别字和专有术语误识等问题。
原文参考是我复制给你的，这位师父讲解的《摩诃止观辅行传弘决辑注》的大致范围。

## 任务：校对逐字稿
结合佛学常识和【原文参考】，纠正转录错别字和专有术语误识等问题。
去掉“嗯”“啊”“呃”“哦”“唉”“这个这个”“那个那个”等显得啰嗦的语气词。
逐字稿所讲的每一点信息（引用、解释、比喻、举例子等等），在输出内容中都要得到完全的保留。

## 输出格式
所有的引经据典（包括引用原文参考），必须用『』括起来。
根据语义自然分段。

【原文参考】
{fuzzy_reference_text}

【逐字稿】
{transcript_text}"""

# 信息丢失检查提示词
verifier_system_prompt = """你是一位非常苛刻的校对专家"""
verifier_user_prompt_template = """
## 背景
逐字稿是一位师父在讲解《摩诃止观辅行传弘决辑注》的录音转录得到的。
校对稿是对逐字稿进行校对得到的，包括修改错别字、专有名词校对、去掉语气词等等。
## 任务
判断逐字稿所讲的每一点信息（引用、解释、比喻、举例子等等），在校对稿中是不是得到了完全的保留。
如果确认没有遗漏信息，直接输出“【无遗漏信息】”。
如果有遗漏，请详细解释遗漏了什么。

【逐字稿】
{transcript_text}

【校对稿】
{written_text}
"""

# 精准定位并提取纯原文（一次调用）
extract_system_prompt = """你是一位严谨的文献整理员。"""
extract_user_prompt_template = """\
## 背景
讲稿是一位师父在讲解《摩诃止观辅行传弘决辑注》。
复制的节选是我估摸着这个讲稿所讲的《摩诃止观辅行传弘决辑注》的大致范围，复制给你看。\
是docx文档节选复制到markdown文件中得到的文本，可能会有些乱码。
复制的节选包括四部分，摩诃止观正文（一般被**包裹住了）、摩诃止观科判（一般跟在正文前面）、\
摩诃止观解释（一般跟在正文后面）、注脚（在整个内容前半标记，在后半罗列注解）
## 思考
思考【讲稿】的内容对应在讲解【复制的节选】中**摩诃止观正文**的哪些句子（按顺序）。
## 任务
把这些摩诃止观正文按顺序直接拼接（不使用换行符等拼接）。

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
        fast_selected_model=FAST_SELECTED_MODEL,
        text_system_prompt=text_system_prompt,
        text_user_prompt_template=text_user_prompt_template,
        verifier_system_prompt=verifier_system_prompt,
        verifier_user_prompt_template=verifier_user_prompt_template,
        extract_system_prompt=extract_system_prompt,
        extract_user_prompt_template=extract_user_prompt_template,
    )
