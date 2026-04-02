import os
from pathlib import Path
from pydub import AudioSegment

from core.asr_engine import Qwen3ASRFlashFiletrans
from core.text_to_text_engine import MiMoText, GeminiText
from core.utils import Color


class AudioProcessingPipeline:
    """端到端音频处理流水线：整段转录 -> 书面化 -> 反向定位精准原文 -> 补充元数据"""

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
            locator_sys_prompt, 
            text_sys_prompt,
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
        print(f"\n{Color.DARK_PURPLE}[1/4] 开始处理整段音频转录...")
        if transcript_filename.exists():
            print(f"{Color.GREEN}✅ 检测到已存在逐字稿，直接复用: {transcript_filename.name}")
            with open(transcript_filename, "r", encoding="utf-8") as f:
                transcript_text = f.read()
        else:
            print(f"{Color.DARK_PURPLE}正在调用 ASR 引擎生成逐字稿...")
            transcript_text = self.asr_engine.recognize(audio_path)

            with open(transcript_filename, "w", encoding="utf-8") as f:
                f.write(transcript_text)
            print(f"{Color.GREEN}逐字稿已保存至: {transcript_filename}")

        # ==========================================
        # 步骤 2: 整理为书面稿 (带缓存)
        # ==========================================
        print(f"\n{Color.DARK_PURPLE}[2/4] 开始处理书面稿...")
        if final_output_filename.exists():
            print(f"{Color.GREEN}✅ 检测到已存在书面稿，直接复用: {final_output_filename.name}")
            with open(final_output_filename, "r", encoding="utf-8") as f:
                written_text = f.read()
        else:
            print(f"{Color.DARK_PURPLE}正在使用 {self.text_engine.model_name} 将逐字稿整理为书面稿，请耐心等待...")
            user_prompt = text_user_prompt_template.format(
                fuzzy_reference_text=fuzzy_reference_text,
                transcript_text=transcript_text
            )

            written_text = self.text_engine.generate(text_sys_prompt, user_prompt)

            with open(final_output_filename, "w", encoding="utf-8") as f:
                f.write(written_text)
            print(f"{Color.GREEN}书面稿已保存至: {final_output_filename}")

        # ==========================================
        # 步骤 3: 反向定位精准原文 (带缓存)
        # ==========================================
        print(f"\n{Color.DARK_PURPLE}[3/4] 开始处理精准原文...")
        if exact_text_filename.exists():
            print(f"{Color.GREEN}✅ 检测到已存在精准原文，直接复用: {exact_text_filename.name}")
            with open(exact_text_filename, "r", encoding="utf-8") as f:
                exact_reference_text = f.read()
        else:
            print(f"{Color.DARK_PURPLE}正在使用 {self.text_engine.model_name} 从书面稿反向定位精准原文...")
            locator_user_prompt = locator_user_prompt_template.format(
                fuzzy_reference_text=fuzzy_reference_text,
                written_text=written_text
            )
            
            exact_reference_text = self.text_engine.generate(locator_sys_prompt, locator_user_prompt)
            
            with open(exact_text_filename, "w", encoding="utf-8") as f:
                f.write(exact_reference_text)
            print(f"{Color.GREEN}精准原文已保存至: {exact_text_filename}")

        # ==========================================
        # 步骤 4: 人工补充元数据并插入书面稿
        # ==========================================
        print(f"\n{Color.DARK_PURPLE}[4/4] 准备补充元数据信息...")
        with open(final_output_filename, "r", encoding="utf-8") as f:
            current_written_text = f.read()
            
        if current_written_text.strip().startswith("> 标题："):
            print(f"{Color.GREEN}✅ 检测到书面稿已包含元数据，跳过元数据输入。")
        else:
            print(f"{Color.RED}请补充以下信息以生成最终文档：")
            year = input("请输入音频的创建时间（年份） [直接回车跳过]: ").strip()
            author = input("请输入音频的作者 [直接回车跳过]: ").strip()
            
            print(f"\n{Color.DARK_PURPLE}提取到的精准原文如下：\n{exact_reference_text}\n")
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
            
            final_content = metadata_header + current_written_text
            with open(final_output_filename, "w", encoding="utf-8") as f:
                f.write(final_content)
            print(f"{Color.GREEN}✅ 元数据已成功插入书面稿开头！")

        print(f"\n{Color.GREEN}===========================================")
        print("🎉 全部流程处理完成！")
        print(f"💡 最终文件列表：")
        print(f"  - 逐字稿: {transcript_filename.name}")
        print(f"  - 书面稿: {final_output_filename.name}")
        print(f"  - 精准原文: {exact_text_filename.name}")
        print(f"===========================================\n")


if __name__ == "__main__":
    # ---------------- 基础配置 ----------------
    audio_path = "摩诃止观-久仁法师/摩诃止观006/摩诃止观006.mp3"
    fuzzy_text_path = "摩诃止观-久仁法师/摩诃止观006/006原文模糊范围.txt"

    # ---------------- 初始化引擎 ----------------
    print(f"{Color.DARK_PURPLE}正在初始化 Qwen3ASRFlashFiletrans 引擎...")
    asr_engine = Qwen3ASRFlashFiletrans(model_name="qwen3-asr-flash-filetrans")
    
    engine_choice = input(f"{Color.DARK_PURPLE}请选择文本处理引擎 (1: Gemini, 2: MiMo) [默认 1]: ").strip()
    if engine_choice == "2":
        print(f"{Color.DARK_PURPLE}正在初始化 MiMoText 引擎...")
        text_engine = MiMoText(model_name="mimo-v2-pro", temperature=0.3)
    else:
        print(f"{Color.DARK_PURPLE}正在初始化 GeminiText 引擎...")
        text_engine = GeminiText(model_name="gemini-3.1-pro-preview", temperature=0.3)

    # ---------------- 系统提示词 ----------------
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
        "你的任务是根据提供的【讲课稿】，在【模糊的原文范围】中截取并输出作者这节课讲到的【原文】。"
        # 信息规范
        "对下节课的预告不需要考虑在截取范围内。"
        "对上节课的回顾不需要考虑在截取范围内。"
        # 输出规范
        "严格从【模糊原文范围】中截取。"
        "只输出截取出的内容。"
    )
    locator_user_prompt_template = (
        "【模糊的原文范围】\n{fuzzy_reference_text}\n\n"
        "【讲课稿】\n{written_text}\n\n"
    )

    # ---------------- 运行流水线 ----------------
    pipeline = AudioProcessingPipeline(
        asr_engine=asr_engine,
        text_engine=text_engine
    )

    pipeline.run(
        audio_path=audio_path,
        fuzzy_text_path=fuzzy_text_path,
        locator_sys_prompt=locator_system_prompt,
        text_sys_prompt=text_system_prompt,
        locator_user_prompt_template=locator_user_prompt_template,
        text_user_prompt_template=text_user_prompt_template
    )
