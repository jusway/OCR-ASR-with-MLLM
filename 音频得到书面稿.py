import os
import threading
from pathlib import Path
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor

from core.audio_chunker import AudioChunker
from core.asr_engine import BaseASR, GeminiASR, MiMoASR
from core.text_to_text_engine import BaseTextModel, GeminiText, MiMoText


class BatchAudioTranscriber:
    """
    批量音频转录器：负责将长音频切片并并发调用 ASR 引擎。
    """

    def __init__(self, asr_engine: BaseASR, chunk_minutes=3, overlap_seconds=10, max_workers=3):
        # 直接注入引擎实例
        self.asr = asr_engine
        self.chunk_minutes = chunk_minutes
        self.overlap_seconds = overlap_seconds
        self.chunker = AudioChunker(chunk_minutes=chunk_minutes, overlap_seconds=overlap_seconds)
        self.max_workers = max_workers
        self.lock = threading.Lock()
        self.completed_count = 0

    def _process_single_chunk(self, chunk_file, chunk_index, system_prompt, prompt, total_chunks):
        chunk_path = Path(chunk_file)
        md_file = chunk_path.with_suffix('.md')

        # 缓存机制：如果已经存在转录好的 md 文件则直接跳过
        if md_file.exists():
            with self.lock:
                self.completed_count += 1
                current_count = self.completed_count
            print(f"--- [缓存] 已完成 {current_count}/{total_chunks}: {md_file.name} ---")
            with open(md_file, "r", encoding="utf-8") as f:
                return f.read(), chunk_file

        # 调用注入的 ASR 引擎的 recognize 方法
        chunk_text = self.asr.recognize(system_prompt, prompt, chunk_file)

        header = f"# 片段 {chunk_index + 1}/{total_chunks}\n\n"
        final_content = header + chunk_text

        with open(md_file, "w", encoding="utf-8") as f:
            f.write(final_content)

        with self.lock:
            self.completed_count += 1
            current_count = self.completed_count
        print(f"--- [进度] 已完成 {current_count}/{total_chunks} 个片段 ---")

        return final_content, chunk_file

    def process_full_audio(self, audio_path, system_prompt, prompt):
        self.completed_count = 0
        audio_path_obj = Path(audio_path)
        chunk_output_dir = audio_path_obj.parent / f"{audio_path_obj.stem}_chunks_{self.chunk_minutes}m"
        os.makedirs(chunk_output_dir, exist_ok=True)

        # 物理切分音频
        chunk_files = self.chunker.process(audio_path, output_dir=str(chunk_output_dir))
        total_chunks = len(chunk_files)

        full_transcription = ""

        # 使用线程池并发执行 ASR 请求
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for i, chunk_file in enumerate(chunk_files):
                future = executor.submit(
                    self._process_single_chunk, chunk_file, i, system_prompt, prompt, total_chunks
                )
                futures.append(future)

            # 按顺序获取结果，保证最终文本逻辑连贯
            for future in futures:
                chunk_text, _ = future.result()
                full_transcription += chunk_text + "\n\n---\n\n"

        return full_transcription

class AudioTextLocator:
    """用于根据完整音频定位原文范围的工具类"""

    def __init__(self, asr_engine: BaseASR):
        self.asr = asr_engine

    def process(self, audio_path: str, text_path: str, system_prompt: str) -> str:
        # 读取模糊范围的原文
        with open(text_path, "r", encoding="utf-8") as f:
            reference_text = f.read()

        prompt = f"【原文模糊范围】\n{reference_text}"

        print(f"正在提交完整音频给模型进行原文定位...")
        # 直接使用传入的系统提示词和完整的音频路径
        location_result = self.asr.recognize(system_prompt, prompt, audio_path)

        return location_result


class AudioProcessingPipeline:
    """端到端音频处理流水线：定位 -> 转录 -> 书面化"""

    def __init__(self, asr_engine: BaseASR, text_engine: BaseTextModel, chunk_minutes=20,
                 overlap_minutes=10, max_workers=8):
        self.asr_engine = asr_engine
        self.text_engine = text_engine
        self.chunk_minutes = chunk_minutes
        self.overlap_minutes = overlap_minutes
        self.max_workers = max_workers

    def _get_audio_duration(self, audio_path):
        audio = AudioSegment.from_mp3(audio_path)
        duration_seconds = int(audio.duration_seconds)
        hours, remainder = divmod(duration_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def run(self, audio_path, fuzzy_text_path, locator_sys_prompt, asr_sys_prompt, text_sys_prompt):
        audio_path_obj = Path(audio_path)
        base_name = audio_path_obj.stem
        parent_dir = audio_path_obj.parent

        located_text_filename = parent_dir / f"{base_name}_原文定位.md"
        transcript_filename = parent_dir / f"{base_name}_逐字稿.md"
        final_output_filename = parent_dir / f"{base_name}_书面稿.md"

        # ==========================================
        # 步骤 1: 原文精准定位
        # ==========================================
        print("\n[1/3] 开始进行原文精准定位...")
        locator = AudioTextLocator(asr_engine=self.asr_engine)
        # 传入外部定义的 locator_sys_prompt
        localized_text = locator.process(audio_path, fuzzy_text_path, locator_sys_prompt)

        with open(located_text_filename, "w", encoding="utf-8") as f:
            f.write(localized_text)
        print(f"原文定位完成！已保存至：{located_text_filename}")

        # ==========================================
        # 步骤 2: 音频切分与逐字稿生成
        # ==========================================
        print("\n[2/3] 开始进行音频切分与逐字稿生成...")
        asr_prompt = f"\n    【原典参考】\n    {localized_text}\n    "

        transcriber = BatchAudioTranscriber(
            asr_engine=self.asr_engine,
            chunk_minutes=self.chunk_minutes,
            overlap_seconds=self.overlap_minutes * 60,
            max_workers=self.max_workers
        )

        transcript_text = transcriber.process_full_audio(audio_path, asr_sys_prompt, asr_prompt)

        with open(transcript_filename, "w", encoding="utf-8") as f:
            f.write(transcript_text)
        print(f"逐字稿已保存至: {transcript_filename}")

        # ==========================================
        # 步骤 3: 整理为书面稿
        # ==========================================
        print(f"\n[3/3] 正在使用 {self.text_engine.model_name} 将逐字稿整理为书面稿，请耐心等待...")
        user_prompt = f"\n【原典参考】\n{localized_text}\n\n【总逐字稿】\n{transcript_text}\n"

        written_text = self.text_engine.generate(text_sys_prompt, user_prompt)

        # ==========================================
        # 步骤 4: 人工补充元数据并生成最终文档
        # ==========================================
        print("\n===========================================")
        print("✅ API 文本处理全部完成！")
        print(f"💡 提示：您可以现在去本地文件夹查看刚生成的中间文件：")
        print(f"  - {located_text_filename.name}")
        print(f"  - {transcript_filename.name}")
        print("参考上述文件内容后，请补充以下信息以生成最终文档：")
        print("===========================================\n")

        year = input("请输入音频的创建时间（年份）: ")
        author = input("请输入音频的作者: ")
        metadata_original_text = input("请输入音频的原文: ")

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

    # ---------------- 选择并初始化 ASR 引擎 ----------------
    print("请选择 ASR (语音识别) 引擎:")
    print("1. Gemini (gemini-3.1-pro-preview)")
    print("2. MiMo (mimo-v2-omni)")
    asr_choice = input("请输入选项 (1 或 2，默认 1): ").strip()
    
    if asr_choice == "2":
        asr_engine = MiMoASR(model_name="mimo-v2-omni", temperature=0.1, top_p=0.95)
    else:
        asr_engine = GeminiASR(model_name="gemini-3.1-pro-preview", temperature=0.1, top_p=0.95)
    print(f"已选择 ASR 引擎: {type(asr_engine).__name__}\n")

    # ---------------- 选择并初始化 Text 引擎 ----------------
    print("请选择 Text (文本处理) 引擎:")
    print("1. Gemini (gemini-3.1-pro-preview)")
    print("2. MiMo (mimo-v2-pro)")
    text_choice = input("请输入选项 (1 或 2，默认 1): ").strip()
    
    if text_choice == "2":
        text_engine = MiMoText(model_name="mimo-v2-pro", temperature=0.3)
    else:
        text_engine = GeminiText(model_name="gemini-3.1-pro-preview", temperature=0.3)
    print(f"已选择 Text 引擎: {type(text_engine).__name__}\n")

    # ---------------- 系统提示词 ----------------
    locator_system_prompt = (
        "你是一位资深的佛法音频校对助手。"
        "听取提供的音频，并阅读提供的【原文模糊范围】。"
        "请精准定位并输出这段音频刚好在讲解【原文模糊范围】中的哪一部分。"
        "只输出这段原文文本。"
        "不要说多余的话。"
    )

    asr_system_prompt = (
        "你是一位资深的佛法音频转录工作的出家比丘大师。"
        "将用户提供的音频准确转录为逐字稿。"
        "保留所有的停顿、口语化表达和重复等所有内容，对音频语音完全忠实。"
        "严格按照用户提供的【原典参考】校对专有名词。"
        "引用原典的时候使用『』符号。"
        "只输出转录内容，不说多余的话。"
    )

    text_system_prompt = (
        "你是一位资深的佛学逐字录音稿整理的比丘师父。"
        "有一个讲法长音频，切分成多个短音频，相邻的短音频互有重叠，分配给了多个义工去做了语音转录，拼接汇总成了总逐字稿。"
        "你的任务是将得到的总逐字稿整理成书面文稿。"
        "风格要大白话、详细，但尽量使再笨的人也能看懂。"
        "讲法者提到的原典文，务必参照【原典参考】进行校对。"
        "引用原典时候使用『』符号框住。"
        "禁止使用md格式。"
        "根据大意自然分段即可。"
        "义工的逐字稿可能会听写有误，遇到可能是谬误的字词，酌情改正。"
        "不丢失任何总逐字稿的信息，也不能自行添加没说的信息。"
        "只输出文稿内容，不说多余的话。"
    )

    # ---------------- 运行流水线 ----------------
    pipeline = AudioProcessingPipeline(
        asr_engine=asr_engine,
        text_engine=text_engine,
        #移除了 target_minutes
        chunk_minutes=10,
        overlap_minutes=5,
        max_workers=12
    )

    pipeline.run(
        audio_path=audio_path,
        fuzzy_text_path=fuzzy_text_path,
        locator_sys_prompt=locator_system_prompt, # 传入定位专用的 Prompt
        asr_sys_prompt=asr_system_prompt,
        text_sys_prompt=text_system_prompt
    )
