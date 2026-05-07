"""纯音频 → 逐字稿 → 校对稿 → 信息丢失检查 → 补充元数据。"""
from pathlib import Path

from pydub import AudioSegment

from core.asr_engine import Qwen3ASRFlashFiletrans
from core.text_engine import DeepSeekText, GeminiText, NvidiaText
from core.utils import Color

# ============================================================
# ================ 以下为全部配置，按需修改 ===================
# ============================================================

# 1) 目标文件夹（脚本自动找 .mp3）
folder_path = Path(r"摩诃止观-定智法师 2017\01通讲001-016\2017年9月25日摩诃止观01通讲001")

# 2) 元数据（写入校对稿头部）
year = "2017"
author = "定智法师"

# 3) ASR 引擎
asr_model_name = "qwen3-asr-flash-filetrans"

# 4) 文本模型
AVAILABLE_MODELS = {
    "kimi":     (NvidiaText, {"model_name": "moonshotai/kimi-k2-instruct-0905", "temperature": 1.0, "top_p": 1.0}),
    "minimax":  (NvidiaText, {"model_name": "minimaxai/minimax-m2.7",           "temperature": 1.0, "top_p": 1.0}),
    "deepseek": (DeepSeekText, {"model_name": "deepseek-v4-pro",                   "temperature": 1.0, "top_p": 0.95}),
    "gemini":   (GeminiText, {"model_name": "gemini-3.1-pro-preview",           "temperature": 1.0}),
}
SELECTED_MODEL = "deepseek"

# 5) 校对提示词
text_system_prompt = "你是一位严谨的转录校对员。"

text_user_prompt_template = """
## 任务
下面是一段未经处理的语音转文字逐字稿，请你把它校订为一份通顺的书面校对稿。
## 逐字稿信息
这是一位师父在讲解摩诃止观。
## 校对规则：
修正错别字：根据上下文和佛学常识纠正同音字和术语误识。
补充标点并根据语义自然分段。
删除口语填充词：去掉"嗯""啊""呃""哦""唉"等语气停顿词；去掉"这个这个""那个那个"等口头禅式反复；去掉句尾不承载语义的"啊""呀""呐""嘛"等拖音。
所有引用都使用『』符号包裹。

【逐字稿】
{transcript_text}"""

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
    audio_files = sorted(folder_path.glob("*.mp3")) + sorted(folder_path.glob("*.MP3"))
    if not audio_files:
        print(f"{Color.RED}错误：在 {folder_path} 下没找到 .mp3 文件")
        exit(1)

    engine_cls, engine_kwargs = AVAILABLE_MODELS[SELECTED_MODEL]
    asr_engine = Qwen3ASRFlashFiletrans(model_name=asr_model_name)
    text_engine = engine_cls(**engine_kwargs)

    for audio_path in audio_files:
        base_name = audio_path.stem
        transcript_path = folder_path / f"{base_name}_逐字稿.md"
        proofread_path = folder_path / f"{base_name}_校对稿.md"

        # 步骤 1: 转录
        if transcript_path.exists():
            print(f"{Color.GREEN}[{base_name}] 逐字稿已存在，跳过转录")
            transcript_text = transcript_path.read_text("utf-8")
        else:
            print(f"{Color.DARK_PURPLE}[{base_name}] 正在转录...")
            transcript_text = asr_engine.recognize(str(audio_path))
            transcript_path.write_text(transcript_text, "utf-8")
            print(f"{Color.GREEN}[{base_name}] 逐字稿已保存")

        # 步骤 2: 校对
        if proofread_path.exists():
            print(f"{Color.GREEN}[{base_name}] 校对稿已存在，跳过校对")
            proofread_text = proofread_path.read_text("utf-8")
        else:
            print(f"{Color.DARK_PURPLE}[{base_name}] 正在校对...")
            user_prompt = text_user_prompt_template.format(transcript_text=transcript_text)
            proofread_text = text_engine.generate(text_system_prompt, user_prompt)
            proofread_path.write_text(proofread_text, "utf-8")
            print(f"{Color.GREEN}[{base_name}] 校对稿已保存")

        # 步骤 3: 信息丢失检查
        verification_path = folder_path / f"{base_name}_丢失信息检查.md"
        if verification_path.exists():
            print(f"{Color.GREEN}[{base_name}] 检查报告已存在，跳过")
        else:
            print(f"{Color.DARK_PURPLE}[{base_name}] 正在检查信息丢失...")
            verifier_prompt = verifier_user_prompt_template.format(
                transcript_text=transcript_text,
                written_text=proofread_text
            )
            report = text_engine.generate(verifier_system_prompt, verifier_prompt)
            verification_path.write_text(report, "utf-8")
            print(f"{Color.GREEN}[{base_name}] 检查报告已保存")

        # 步骤 4: 补充元数据
        if not proofread_text.strip().startswith("> 标题："):
            duration_seconds = int(AudioSegment.from_mp3(str(audio_path)).duration_seconds)
            h, m, s = duration_seconds // 3600, (duration_seconds % 3600) // 60, duration_seconds % 60
            duration = f"{h:02d}:{m:02d}:{s:02d}"
            metadata = f"> 标题：{base_name}\n> 时间：{year}\n> 时长：{duration}\n> 作者：{author}\n\n---\n\n"
            proofread_text = metadata + proofread_text
            proofread_path.write_text(proofread_text, "utf-8")
            print(f"{Color.GREEN}[{base_name}] 元数据已插入校对稿")

    print(f"\n{Color.GREEN}全部完成。")
