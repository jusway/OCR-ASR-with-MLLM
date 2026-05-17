from pathlib import Path

from core.pipeline import run
from core.text_engine import DeepSeekText, GeminiText, NvidiaText

# ============================================================
# ================ 以下为全部配置，按需修改 ===================
# ============================================================

# 1) 任务文件夹（脚本自动找 .mp3）
folder_path = Path(r"摩诃止观-定智法师 2017/03正说分第一大意发大心/摩诃止观01正说分/2017年12月25日摩诃止观01正说分第一大意发大心032")

# 2) 元数据（写入校对稿头部）
year = "2017"
author = "定智法师"

# 3) ASR 引擎模型名
asr_model_name = "qwen3-asr-flash-filetrans"

# 4) 文本模型：想换模型只改 SELECTED_MODEL 这一行的 key
AVAILABLE_MODELS = {
    "deepseek": (DeepSeekText, {                          # 前3步：长思考
        "model_name": "deepseek-v4-pro",
        "enable_thinking": True, "reasoning_effort": "max",
    }),
    "deepseek-fast": (DeepSeekText, {                     # 后2步：无思考
        "model_name": "deepseek-v4-flash",
        "enable_thinking": False,
    }),
}
SELECTED_MODEL = "deepseek"
FAST_SELECTED_MODEL = "deepseek-fast"

# 模糊原文种子（首次运行时自动创建 模糊原文.md，此后以文件内容为准）
reference_text = ""

# 逐字稿+大致原文=得到校对稿
text_system_prompt = '你是一位严谨的校对员。'
text_user_prompt_template = """\
## 背景
逐字稿是一位师父在讲解《摩诃止观辅行传弘决辑注》
原文参考是我复制给你的，这位师父讲解《摩诃止观辅行传弘决辑注》的大致范围。
## 任务
校对逐字稿。
## 校对规则：
修正错别字：结合佛学常识和【原文参考】，纠正同音字和专有术语误识。
补充标点并根据语义自然分段。
去掉“嗯”“啊”“呃”“哦”“唉”等语气停顿词。
去掉“这个这个”“那个那个”等口语过渡词。
讲解中所有的引经据典（包括引用原文参考），必须用『』括起来。
## 重要
思考逐字稿讲的每一个知识点，然后保证每一个知识点的信息在输出内容中都得到完全的保留。

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
思考逐字稿讲的每一个知识点是什么，然后思考每一个知识点的信息在校对稿中是否得到了完全的保留。
如果确认没有遗漏信息，直接输出“【无遗漏信息】”。
如果有遗漏，请详细解释遗漏了什么。

【逐字稿】{transcript_text}
【校对稿】{written_text}
"""

# 定位精准原文
precision_system_prompt = '你是一位严谨的校对员。'
precision_user_prompt_template = """\
## 背景
校对稿是一位师父在讲解《摩诃止观辅行传弘决辑注》
原文大致范围是我估摸着这个校对稿所讲的《摩诃止观辅行传弘决辑注》的大致范围，复制给你看。
原文大致范围前半部分是正文，后半部分（如果有的话）是对正文的注脚。
## 任务
你的任务是思考【校对稿】的内容对应【原文大致范围】（前半部分），从哪里开始讲，到哪里结束讲解。
然后输出且仅输出这个开始到结束的内容。
也就是说，我想要你定位校对稿到底讲解了原文大致范围的哪一段内容。

【校对稿】
{written_text}
【原文大致范围】
{fuzzy_reference_text}
"""

# 从精准原文中提取纯原文
extraction_system_prompt = """你是一位严谨的文献整理员。"""
extraction_user_prompt_template = """\
## 背景
所给的文献是《摩诃止观辅行传弘决辑注》的docx文档节选，复制到markdown文件中得到的文本。
可能会有些乱码。
## 任务
请从文献中提取出纯粹的摩诃止观正文输出。
正文按顺序直接拼接就好，不使用换行符等等。
一般来说摩诃止观正文被**包裹，例如：**若能如此简非显是，体权识实而发心者，是一切诸佛种。**
## 举例子
输入：
△五约譬广叹二，初约理以明生善灭恶德二，初正释二，初明生善德
**譬如金刚，从金性生，佛菩提心，从大悲起，是诸行先；如服阿娑罗药，先用清水，诸行中最；如诸根中命根为最，佛正法正行中此心为最；如太子生，具王仪相，大臣恭敬，有大声名；如迦陵频伽鸟，****[****谷****-****禾****+****卵****]****中鸣声**[***\*[1\]\****](#_ftn1)**，已胜诸鸟。**
譬如下，举十譬叹德，此即约理以叹生善灭恶德也，以菩提心皆依理故。如金刚等五，生善德也。如师子弦等五，灭恶德也。
...
△二明灭恶德
**此菩提心有大势力，如师子筋弦**[***\*[7\]\****](#_ftn7)**；如师子乳**[***\*[8\]\****](#_ftn8)**；如金刚槌；如那罗延箭；具足众宝，能除贫苦，如如意珠。虽小懈怠，小失威仪，犹胜二乘功德。**
次灭恶中，奏圆教弦，偏弦断绝。奏者，为也，凡为乐音，皆称为奏。

输出：
譬如金刚，从金性生，佛菩提心，从大悲起，是诸行先；如服阿娑罗药，先用清水，诸行中最；如诸根中命根为最，佛正法正行中此心为最；如太子生，具王仪相，大臣恭敬，有大声名；如迦陵频伽鸟，[谷-禾+卵]中鸣声 ，已胜诸鸟。此菩提心有大势力，如师子筋弦 ；如师子乳 ；如金刚槌；如那罗延箭；具足众宝，能除贫苦，如如意珠。虽小懈怠，小失威仪，犹胜二乘功德。

【文献】
{precision_text}
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
        precision_system_prompt=precision_system_prompt,
        precision_user_prompt_template=precision_user_prompt_template,
        extraction_system_prompt=extraction_system_prompt,
        extraction_user_prompt_template=extraction_user_prompt_template,
    )
