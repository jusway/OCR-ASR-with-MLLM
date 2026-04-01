import os
import concurrent.futures
from pathlib import Path
from pydub import AudioSegment

from core.asr_engine import GeminiASR
from core.text_to_text_engine import GeminiText
from core.utils import AudioChunker


class AudioProcessingPipeline:
    """旧版端到端音频处理流水线：切片 -> 并发转录 -> 拼接 -> 书面化"""

    def __init__(self, asr_engine, text_engine, chunk_minutes=20, overlap_minutes=10, max_workers=8):
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

    def run(self, audio_path, prompt_path, asr_sys_prompt, text_sys_prompt):
        audio_path_obj = Path(audio_path)
        base_name = audio_path_obj.stem
        parent_dir = audio_path_obj.parent

        transcript_filename = parent_dir / f"{base_name}_逐字稿.md"
        final_output_filename = parent_dir / f"{base_name}_书面稿.md"

        # 读取原文
        with open(prompt_path, "r", encoding="utf-8") as f:
            reference_text = f.read()

        # ==========================================
        # 步骤 1: 音频切片
        # ==========================================
        print("\n[1/3] 正在进行音频切片...")
        chunker = AudioChunker(chunk_minutes=self.chunk_minutes, overlap_seconds=self.overlap_minutes * 60)
        chunks_dir = str(parent_dir / "temp_chunks")
        chunk_files = chunker.process(audio_path, output_dir=chunks_dir)
        print(f"音频已切分为 {len(chunk_files)} 个片段。")

        # ==========================================
        # 步骤 2: 并发转录
        # ==========================================
        print("\n[2/3] 开始并发转录音频切片...")
        asr_prompt = f"【原典参考】\n{reference_text}\n"
        
        transcripts = [""] * len(chunk_files)
        
        def process_chunk(index, chunk_file):
            print(f"  -> 正在处理切片 {index + 1}/{len(chunk_files)}: {os.path.basename(chunk_file)}")
            text = self.asr_engine.recognize(asr_sys_prompt, asr_prompt, chunk_file)
            return index, text

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(process_chunk, i, f) for i, f in enumerate(chunk_files)]
            for future in concurrent.futures.as_completed(futures):
                idx, text = future.result()
                transcripts[idx] = text

        transcript_text = "\n".join(transcripts)

        with open(transcript_filename, "w", encoding="utf-8") as f:
            f.write(transcript_text)
        print(f"逐字稿已拼接并保存至: {transcript_filename}")

        # ==========================================
        # 步骤 3: 整理为书面稿
        # ==========================================
        print(f"\n[3/3] 正在将逐字稿整理为书面稿，请耐心等待...")
        user_prompt = f"【原典参考】\n{reference_text}\n\n【总逐字稿】\n{transcript_text}\n"

        written_text = self.text_engine.generate(text_sys_prompt, user_prompt)

        # ==========================================
        # 步骤 4: 补充元数据
        # ==========================================
        print("\n===========================================")
        print("✅ API 文本处理全部完成！")
        
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
    audio_path = "../摩诃止观-久仁法师/摩诃止观001/摩诃止观001.mp3"
    prompt_path = "../摩诃止观-久仁法师/摩诃止观001/001原文模糊范围.txt"

    # ---------------- 初始化引擎 ----------------
    print("正在初始化 GeminiASR 引擎...")
    asr_engine = GeminiASR(model_name="gemini-3.1-pro-preview", temperature=0.1, top_p=0.95)
    
    print("正在初始化 GeminiText 引擎...")
    text_engine = GeminiText(model_name="gemini-3.1-pro-preview", temperature=0.3)

    # ---------------- 系统提示词 ----------------
    asr_system_prompt = (
        "您是一位资深的佛法音频转录工作的出家比丘大师。"
        "将用户提供的音频准确转录为逐字稿。"
        "保留所有的停顿、口语化表达和重复等所有内容，对音频语音完全忠实。"
        "严格按照用户提供的【原典参考】校对专有名词。"
        "引用原典的时候使用『』符号。"
        "只输出转录内容，不说多余的话。"
    )

    text_system_prompt = (
        "您是一位资深的佛学逐字录音稿整理的比丘师父。"
        "您的任务是将得到的讲法长音频逐字稿整理成书面文稿。"
        "风格要大白话、详细，但尽量使再笨的人也能看懂。"
        "讲法者提到的原典文，务必参照【原典参考】进行校对。"
        "引用原典时候使用『』符号框住。"
        "禁止使用md格式。"
        "根据大意自然分段即可。"
        "逐字稿可能会听写有误，遇到可能是谬误的字词，酌情改正。"
        "不丢失任何总逐字稿的信息，也不能自行添加没说的信息。"
        "只输出文稿内容，不说多余的话。"
    )

    # ---------------- 运行流水线 ----------------
    pipeline = AudioProcessingPipeline(
        asr_engine=asr_engine,
        text_engine=text_engine,
        chunk_minutes=20,
        overlap_minutes=10,
        max_workers=8
    )

    pipeline.run(
        audio_path=audio_path,
        prompt_path=prompt_path,
        asr_sys_prompt=asr_system_prompt,
        text_sys_prompt=text_system_prompt
    )
