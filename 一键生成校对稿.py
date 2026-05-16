from pathlib import Path

from core.pipeline import run
from core.text_engine import DeepSeekText, GeminiText, NvidiaText

# ============================================================
# ================ 以下为全部配置，按需修改 ===================
# ============================================================

# 1) 任务文件夹（脚本自动找 .mp3）
folder_path = Path(r"摩诃止观-定智法师 2017/03正说分第一大意发大心/摩诃止观01正说分/2017年12月19日摩诃止观01正说分第一大意发大心025")

# 2) 元数据（写入校对稿头部）
year = "2017"
author = "定智法师"

# 3) ASR 引擎模型名
asr_model_name = "qwen3-asr-flash-filetrans"

# 4) 文本模型：想换模型只改 SELECTED_MODEL 这一行的 key
AVAILABLE_MODELS = {
    "deepseek": (DeepSeekText, {"model_name": "deepseek-v4-pro",                   "temperature": 1.0, "top_p": 1.0}),
}
SELECTED_MODEL = "deepseek"

# 模糊原文种子（首次运行时自动创建 模糊原文.md，此后以文件内容为准）
reference_text = ""

# 逐字稿+大致原文=得到校对稿
text_system_prompt = '你是一位严谨的校对员。'
text_user_prompt_template = """\
## 背景
逐字稿是一位师父在讲解《摩诃止观辅行传弘决辑注》
原文参考是我复制给你的，这位师父讲解《摩诃止观辅行传弘决辑注》的大致范围。
## 任务
按照且仅按照校对规则，来校对逐字稿。
## 校对规则：
修正错别字：结合佛学常识和【原文参考】，纠正同音字和术语误识。
补充标点并根据语义自然分段。
去掉“嗯”“啊”“呃”“哦”“唉”等语气停顿词；
去掉“这个这个”“那个那个”等口语过渡词；
讲解中引用【原文参考】词句时，必须用『』括起来。

【原文参考】
{fuzzy_reference_text}
【逐字稿】
{transcript_text}"""

# 信息丢失检查提示词
verifier_system_prompt = """你是一位非常苛刻的校对专家"""
verifier_user_prompt_template = """
## 背景
逐字稿是一位师父在讲解《摩诃止观辅行传弘决辑注》。
校对稿是对逐字稿进行校对得到的，包括修改错别字、专有名词校对、去掉语气词等等。
## 任务
你的任务是检查校对稿是否丢失了逐字稿的关键信息。
如果没有遗漏，直接输出【无遗漏信息】。
如果有遗漏，请解释遗漏了什么。
【逐字稿】{transcript_text}
【校对稿】{written_text}
"""

# 定位精准原文
precision_system_prompt = '你是一位严谨的校对员。'
precision_user_prompt_template = """\
## 背景
校对稿是一位师父在讲解《摩诃止观辅行传弘决辑注》
原文大致范围是我估摸着这个校对稿所讲的《摩诃止观辅行传弘决辑注》的大致范围，复制给你看。
## 任务
你的任务是思考【校对稿】的内容对应【原文大致范围】中从哪里开始，到哪里结束。
然后原封不动地输出【原文大致范围】从这个开始到结束的全部内容，包括格式也要一模一样。
再次强调，格式也要一模一样！

【校对稿】
{written_text}
【原文大致范围】
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
        text_system_prompt=text_system_prompt,
        text_user_prompt_template=text_user_prompt_template,
        verifier_system_prompt=verifier_system_prompt,
        verifier_user_prompt_template=verifier_user_prompt_template,
        precision_system_prompt=precision_system_prompt,
        precision_user_prompt_template=precision_user_prompt_template,
    )
