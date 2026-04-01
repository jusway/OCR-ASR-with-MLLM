import os
from pathlib import Path
from pydub import AudioSegment

from core.asr_engine import MiMoASR
from core.text_to_text_engine import MiMoText


def clean_repeated_text(text: str) -> str:
    """
    简单的后处理：移除连续重复的相同行（常见于ASR在静音处的幻觉）。
    如果某一行与上一行完全相同，则跳过不输出。
    """
    lines = text.split('\n')
    if not lines:
        return text
    
    cleaned = []
    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            cleaned.append(line)
            continue
            
        if cleaned and stripped_line == cleaned[-1].strip():
            continue
            
        cleaned.append(line)
        
    return '\n'.join(cleaned)


class AudioProcessingPipeline:
    """端到端音频处理流水线：整段转录 -> 定位精准原文 -> 书面化"""

    def __init__(self, asr_engine, text_engine):
        self.asr_engine = asr_engine
        self.text_engine = text_engine

    def _get_audio_duration(self, audio_path):
        audio = AudioSegment.from_mp3(audio_path)
        duration_seconds = int(audio.duration_seconds)
        hours, remainder = divmod(duration_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def run(self, audio_path, fuzzy_text_path, asr_sys_prompt, locator_sys_prompt, text_sys_prompt):
        audio_path_obj = Path(audio_path)
        base_name = audio_path_obj.stem
        parent_dir = audio_path_obj.parent

        transcript_filename = parent_dir / f"{base_name}_逐字稿.md"
        exact_text_filename = parent_dir / f"{base_name}_精准原文.txt"
        final_output_filename = parent_dir / f"{base_name}_书面稿.md"

        # 读取模糊范围的原文
        with open(fuzzy_text_path, "r", encoding="utf-8") as f:
            fuzzy_reference_text = f.read()

        # ==========================================
        # 步骤 1: 整段音频直接转录
        # ==========================================
        print("\n[1/3] 开始进行整段音频转录...")
        asr_prompt = f"【原典参考】\n{fuzzy_reference_text}\n"
        
        raw_transcript = self.asr_engine.recognize(asr_sys_prompt, asr_prompt, audio_path)
        
        # 过滤可能出现的复读机幻觉
        transcript_text = clean_repeated_text(raw_transcript)

        with open(transcript_filename, "w", encoding="utf-8") as f:
            f.write(transcript_text)
        print(f"逐字稿已保存至: {transcript_filename}")

        # ==========================================
        # 步骤 2: 定位精准原文
        # ==========================================
        print(f"\n[2/3] 正在使用 {self.text_engine.model_name} 从模糊原文中定位精准原文...")
        locator_user_prompt = (
            f"【模糊原文范围】\n{fuzzy_reference_text}\n\n"
            f"【语音逐字稿】\n{transcript_text}\n\n"
            "请根据逐字稿的内容，从模糊原文中提取出讲法者实际讲到的精准原文范围。"
        )
        
        exact_reference_text = self.text_engine.generate(locator_sys_prompt, locator_user_prompt)
        
        with open(exact_text_filename, "w", encoding="utf-8") as f:
            f.write(exact_reference_text)
        print(f"精准原文已保存至: {exact_text_filename}")

        # ==========================================
        # 步骤 3: 整理为书面稿
        # ==========================================
        print(f"\n[3/3] 正在使用 {self.text_engine.model_name} 将逐字稿整理为书面稿，请耐心等待...")
        # 这里使用刚刚提取出的精准原文作为参考
        user_prompt = f"【原典参考】\n{exact_reference_text}\n\n【总逐字稿】\n{transcript_text}\n"

        written_text = self.text_engine.generate(text_sys_prompt, user_prompt)

        # ==========================================
        # 步骤 4: 人工补充元数据并生成最终文档
        # ==========================================
        print("\n===========================================")
        print("✅ API 文本处理全部完成！")
        print(f"💡 提示：您可以现在去本地文件夹查看刚生成的中间文件：")
        print(f"  - {transcript_filename.name}")
        print(f"  - {exact_text_filename.name}")
        print("参考上述文件内容后，请补充以下信息以生成最终文档：")
        print("===========================================\n")

        year = input("请输入音频的创建时间（年份） [直接回车跳过]: ").strip()
        author = input("请输入音频的作者 [直接回车跳过]: ").strip()
        
        print(f"\n提取到的精准原文如下：\n{exact_reference_text}\n")
        metadata_original_text = input("请输入音频的原文 [直接回车跳过]: ").strip()

        duration = self._get_audio_duration(audio_path)
        metadata_header = (
            f"> 标题：{base_name}\n"
            f"> 时间：{year}\n"
            f"> 时长：{duration}\n"
            f"> 作者：{author}\n"
            f"> 原文：{metadata_original_text}\n"
            "\n"
            "---\n\n"
        )

        final_content = metadata_header + written_text

        with open(final_output_filename, "w", encoding="utf-8") as f:
            f.write(final_content)

        print(f"\n🎉 全部完成！带元数据的书面稿已成功写入：{final_output_filename}")


if __name__ == "__main__":
    # ---------------- 基础配置 ----------------
    audio_path = "摩诃止观-久仁法师/摩诃止观001/摩诃止观001.mp3"
    fuzzy_text_path = "摩诃止观-久仁法师/摩诃止观001/001原文模糊范围.txt"

    # ---------------- 初始化引擎 ----------------
    print("正在初始化 MiMoASR 引擎...")
    asr_engine = MiMoASR(model_name="mimo-v2-omni", temperature=0.1, top_p=0.95)
    
    print("正在初始化 MiMoText 引擎...")
    text_engine = MiMoText(model_name="mimo-v2-pro", temperature=0.3)

    # ---------------- 系统提示词 ----------------
    asr_system_prompt = (
        "您是一位资深的佛法音频转录工作的出家比丘大师。"
        "将用户提供的音频准确转录为逐字稿。"
        "保留所有的停顿、口语化表达和重复等所有内容，对音频语音完全忠实。"
        "严格按照用户提供的【原典参考】校对专有名词。"
        "引用原典的时候使用『』符号。"
        "只输出转录内容，不说多余的话。"
        "【重要警告】：如果遇到音频尾部长时间静音、无意义的背景音或听不清的地方，请直接停止转录，**绝对不要**自行脑补、幻觉或反复输出相同的句子。请务必只转录实际听到的清晰语音！"
    )

    locator_system_prompt = (
        "你是一个严谨的校对专家。"
        "你的任务是根据提供的【语音逐字稿】，在【模糊原文范围】中找出讲法者真正讲到的【精准原文】。"
        "请严格从【模糊原文范围】中截取，逐字稿中就算提了一下，也要把这段原文囊括到输出结果中。"
        "只输出截取出的精准原文内容。"
    )

    text_system_prompt = (
        # 角色
        "您是一位资深的佛学逐字录音稿整理的比丘师父。"
        # 任务
        "您的任务是将得到的讲法长音频逐字稿整理成书面文稿。"
        # 要求注意
        "风格要大白话、详细，但尽量使再笨的人也能看懂。"
        "讲法者提到的原典文，务必参照【原典参考】进行校对。"
        "引用原典时候使用『』符号框住。"
        "禁止使用md格式。"
        "根据大意自然分段即可。"
        # 信息规范
        "逐字稿可能会听写有误，遇到可能是谬误的字词，酌情改正。"
        "不丢失任何总逐字稿的信息，也不能自行添加没说的信息。"
        "只输出文稿内容，不说多余的话。"
    )

    # ---------------- 运行流水线 ----------------
    pipeline = AudioProcessingPipeline(
        asr_engine=asr_engine,
        text_engine=text_engine
    )

    pipeline.run(
        audio_path=audio_path,
        fuzzy_text_path=fuzzy_text_path,
        asr_sys_prompt=asr_system_prompt,
        locator_sys_prompt=locator_system_prompt,
        text_sys_prompt=text_system_prompt
    )
