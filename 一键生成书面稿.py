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
    """端到端音频处理流水线：整段转录 -> 书面化 -> 反向定位精准原文"""

    def __init__(self, asr_engine, text_engine):
        self.asr_engine = asr_engine
        self.text_engine = text_engine

    def _get_audio_duration(self, audio_path):
        audio = AudioSegment.from_mp3(audio_path)
        duration_seconds = int(audio.duration_seconds)
        hours, remainder = divmod(duration_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def run(self, 
            audio_path, 
            fuzzy_text_path, 
            asr_sys_prompt, 
            locator_sys_prompt, 
            text_sys_prompt,
            asr_user_prompt_template,
            locator_user_prompt_template,
            text_user_prompt_template):
        
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
        # 步骤 1: 整段音频直接转录 (带缓存)
        # ==========================================
        print("\n[1/4] 开始处理整段音频转录...")
        if transcript_filename.exists():
            print(f"✅ 检测到已存在逐字稿，直接复用: {transcript_filename.name}")
            with open(transcript_filename, "r", encoding="utf-8") as f:
                transcript_text = f.read()
        else:
            print("正在调用 ASR 引擎生成逐字稿...")
            asr_prompt = asr_user_prompt_template.format(fuzzy_reference_text=fuzzy_reference_text)
            raw_transcript = self.asr_engine.recognize(asr_sys_prompt, asr_prompt, audio_path)
            
            # 过滤可能出现的复读机幻觉
            transcript_text = clean_repeated_text(raw_transcript)

            with open(transcript_filename, "w", encoding="utf-8") as f:
                f.write(transcript_text)
            print(f"逐字稿已保存至: {transcript_filename}")

        # ==========================================
        # 步骤 2: 人工补充元数据
        # ==========================================
        print("\n[2/4] 准备生成书面稿，请先补充元数据信息：")
        if final_output_filename.exists():
            print("✅ 检测到已存在书面稿，跳过元数据输入。")
        else:
            year = input("请输入音频的创建时间（年份） [直接回车跳过]: ").strip()
            author = input("请输入音频的作者 [直接回车跳过]: ").strip()
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

        # ==========================================
        # 步骤 3: 整理为书面稿 (带缓存)
        # ==========================================
        print("\n[3/4] 开始处理书面稿...")
        if final_output_filename.exists():
            print(f"✅ 检测到已存在书面稿，直接复用: {final_output_filename.name}")
            with open(final_output_filename, "r", encoding="utf-8") as f:
                written_text = f.read()
        else:
            print(f"正在使用 {self.text_engine.model_name} 将逐字稿整理为书面稿，请耐心等待...")
            user_prompt = text_user_prompt_template.format(
                fuzzy_reference_text=fuzzy_reference_text,
                transcript_text=transcript_text
            )

            generated_written_text = self.text_engine.generate(text_sys_prompt, user_prompt)
            written_text = metadata_header + generated_written_text

            with open(final_output_filename, "w", encoding="utf-8") as f:
                f.write(written_text)
            print(f"书面稿已保存至: {final_output_filename}")

        # ==========================================
        # 步骤 4: 反向定位精准原文 (带缓存)
        # ==========================================
        print("\n[4/4] 开始处理精准原文...")
        if exact_text_filename.exists():
            print(f"✅ 检测到已存在精准原文，直接复用: {exact_text_filename.name}")
            with open(exact_text_filename, "r", encoding="utf-8") as f:
                exact_reference_text = f.read()
        else:
            print(f"正在使用 {self.text_engine.model_name} 从书面稿反向定位精准原文...")
            locator_user_prompt = locator_user_prompt_template.format(
                fuzzy_reference_text=fuzzy_reference_text,
                written_text=written_text
            )
            
            exact_reference_text = self.text_engine.generate(locator_sys_prompt, locator_user_prompt)
            
            with open(exact_text_filename, "w", encoding="utf-8") as f:
                f.write(exact_reference_text)
            print(f"精准原文已保存至: {exact_text_filename}")

        print("\n===========================================")
        print("🎉 全部流程处理完成！")
        print(f"💡 最终文件列表：")
        print(f"  - 逐字稿: {transcript_filename.name}")
        print(f"  - 书面稿: {final_output_filename.name}")
        print(f"  - 精准原文: {exact_text_filename.name}")
        print("===========================================\n")


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
        # 角色
        "您是一位资深的佛法音频转录工作的出家比丘大师。"
        # 任务
        "将用户提供的音频准确转录为逐字稿。"
        # 信息规范
        "保留所有的停顿、口语化表达和重复等所有内容，对音频语音完全忠实。"
        "【重要警告】：如果遇到音频尾部长时间静音、无意义的背景音或听不清的地方，请直接停止转录，**绝对不要**自行脑补、幻觉或反复输出相同的句子。请务必只转录实际听到的清晰语音！"
        # 输出规范
        "严格按照用户提供的【原文参考】校对专有名词。"
        "【强制格式要求】：当转录到【原文参考】中的原话时，**必须且只能**使用『』符号将其引起来（例如：『如是我闻』）。绝对不要使用普通的双引号！"
        "只输出转录内容，不说多余的话。"
    )
    asr_user_prompt_template = "【原文参考】\n{fuzzy_reference_text}\n"


    text_system_prompt = (
        # 角色
        "您是一位精通佛学逐字录音稿整理的比丘师父。"
        # 任务
        "您的任务是将讲法长音频转录得到的【逐字稿】整理成书面文稿。"
        # 信息规范
        "【逐字稿】可能会听写有误，遇到可能是谬误的字词，酌情改正。"
        "不丢失任何【逐字稿】的信息，也不能自行添加没说的信息。"
        "只输出文稿内容，不说多余的话。"
        # 输出规范
        "风格要大白话、详细，但尽量使再笨的人也能看懂。"
        "参照【原文参考】校对专有名词。"
        "禁止使用md格式。"
        "根据大意自然分段即可。"
        "【强制格式要求】：输出的时候，只要出现了【原文参考】中的内容，**必须且只能**使用『』符号将其引起来（例如：师父说『如是我闻』）。绝对不要使用普通的双引号“”！"
    )
    text_user_prompt_template = (
        "【原文参考】\n{fuzzy_reference_text}\n\n"
        "【逐字稿】\n{transcript_text}\n"
    )

    locator_system_prompt = (
        # 角色
        "你是一个严谨的校对专家。"
        # 任务
        "你的任务是根据提供的【书面稿】，在【模糊原文范围】中找出作者讲到的【原文】。"
        # 输出规范
        "请严格从【模糊原文范围】中截取，【书面稿】中就算仅仅提了一下，也要把这段原文囊括到输出结果中。"
        "只输出截取出的精准原文内容。"
    )
    locator_user_prompt_template = (
        "【模糊原文范围】\n{fuzzy_reference_text}\n\n"
        "【书面稿】\n{written_text}\n\n"
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
        text_sys_prompt=text_system_prompt,
        asr_user_prompt_template=asr_user_prompt_template,
        locator_user_prompt_template=locator_user_prompt_template,
        text_user_prompt_template=text_user_prompt_template
    )
