from pathlib import Path

from core.pipeline import run
from core.text_engine import DeepSeekText, GeminiText, NvidiaText

# ============================================================
# ================ 以下为全部配置，按需修改 ===================
# ============================================================

# 1) 任务文件夹（脚本会自动从中找 .mp3 与原文 .txt）
folder_path = Path("摩诃止观-久仁法师/摩诃止观041")

# 2) 元数据（写入校对稿头部）
year = "null"
author = "久仁法师"
reference_text = """\
又圣说多端，或次说或不次说，或具说或不具说，或杂说或不杂说。众生禀益不同，或次益不次益，或具益不具益，或杂益不杂益。或四悉檀成五缘，五缘成四悉；或四悉成一因缘，一因缘成一悉；或一一因缘皆具四悉，四悉具五缘。如是等种种互相成显，还以三止观结之，可以意知。又以一止观结之，发菩提心即是观，邪僻心息即是止。又五略只是十广，初五章只是发菩提心一意耳；方便、正观只是四三昧耳；果报一章只明违顺，违即二边果报，顺即胜妙果报；起教一章，转其自心利益于他，或作佛身施权实，或作九界像对扬渐顿、转渐顿、弘通渐顿；旨归章只是同归大处，秘密藏中，故知略广意同也。
"""

# 3) ASR 引擎模型名
asr_model_name = "qwen3-asr-flash-filetrans"

# 4) 文本模型：想换模型只改 SELECTED_MODEL 这一行的 key
#    新增模型：照样追加一行 "别名": (引擎类, {"model_name": "...", ...其它初始化参数}),
AVAILABLE_MODELS = {
    "kimi":     (NvidiaText, {"model_name": "moonshotai/kimi-k2-instruct-0905", "temperature": 1.0, "top_p": 1.0}),
    "minimax":  (NvidiaText, {"model_name": "minimaxai/minimax-m2.7",           "temperature": 1.0, "top_p": 1.0}),
    "deepseek": (DeepSeekText, {"model_name": "deepseek-v4-pro",                   "temperature": 1.0, "top_p": 0.95}),
    "gemini":   (GeminiText, {"model_name": "gemini-3.1-pro-preview",           "temperature": 1.0}),
}
SELECTED_MODEL = "deepseek"

# 5) 书面化提示词
text_system_prompt = """\
【你的任务】
你是严谨的转录校对员。你的任务是把 ASR 识别有误的口语逐字稿，修正为准确、通顺、自然分段的书面文字。

【只允许做三件事】
1. 修正错别字：结合佛学常识和【原文参考】，纠正同音字和术语误识（例：“一官俱习”→“异根俱益”），拿不准的保留原字。
2. 补充标点并分段：补上缺失的句读；根据语义将文字划分为自然段落，段间空一行，不编号不加小标题。每段 200-600 字。
3. 删除口语填充词：去掉“嗯”“啊”“呃”“哦”“唉”等语气停顿词；去掉“这个这个”“那个那个”等口头禅式反复；去掉句尾不承载语义的“啊”“呀”“呐”“嘛”等拖音。但“这个法门”中的“这个”是实义词，必须保留。

【分段时机】
话题转换
逻辑条目（“第一……第二……第三……”“首先……其次……”等逐条展开处）
明显过渡语（“好，接下来……”“再来看……”“以上是……”等承接处）
等等

【引文与分段规范】
1. 讲解中引用【原文参考】词句时，必须用『』括起来，不得用普通引号代替。
2. 原文参考中有明显段落结构的，优先参考其分段方式。
"""

text_user_prompt_template = """\
下面是你要校对的逐字稿。请记住：不要删减任何内容，并在每个话题或逻辑转换处空行分段。

【原文参考】
{fuzzy_reference_text}

【逐字稿】
{transcript_text}

请从头到尾校对全文，修正错别字和标点，并在语义切换处空行分段。
"""

# 6) 信息丢失检查提示词
verifier_system_prompt = (
    "你是一位严谨校对员。你的任务是对比【原始逐字稿】和【校对稿】，检查校对稿是否丢失了逐字稿中的关键信息。"
    "如果没有遗漏，直接输出【无遗漏信息】。"
    "如果有遗漏，请解释遗漏了什么。"
)
verifier_user_prompt_template = (
    "【原始逐字稿】\n{transcript_text}\n\n"
    "【校对稿】\n{written_text}\n"
)

# ============================================================
# ================ 执行 ========================================
# ============================================================
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
    )
