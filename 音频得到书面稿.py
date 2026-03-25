import os
import uuid
from pathlib import Path
from pydub import AudioSegment

# 假设这些是你核心库的导入
from core.asr import GeminiASR
from core.batch_audio_transcriber import BatchAudioTranscriber
from core.gemini_text import GeminiText


class AudioTailTextLocator:
    """用于根据音频尾部定位原文范围的工具类"""

    def __init__(self, model_name, tail_minutes=20, temperature=0.1, top_p=0.95):
        self.asr = GeminiASR(temperature=temperature, top_p=top_p, model_name=model_name)
        self.tail_minutes = tail_minutes

    def _extract_tail(self, audio_path: str) -> str:
        audio = AudioSegment.from_file(audio_path)
        tail_ms = self.tail_minutes * 60 * 1000

        if len(audio) < tail_ms:
            chunk = audio
        else:
            chunk = audio[-tail_ms:]

        out_path = f"temp_tail_{uuid.uuid4().hex}.mp3"
        chunk.export(out_path, format="mp3")
        return out_path

    def process(self, audio_path: str, text_path: str) -> str:
        temp_audio_path = self._extract_tail(audio_path)

        with open(text_path, "r", encoding="utf-8") as f:
            reference_text = f.read()

        system_prompt = (
            "你是一位资深的佛法音频校对助手。"
            "听取提供的音频，并阅读提供的【原文模糊范围】。"
            "请精准定位并输出这段音频刚好在讲解【原文模糊范围】中的哪一部分。"
            "只输出这段原文文本。"
            "不要说多余的话。"
        )

        prompt = f"【原文模糊范围】\n{reference_text}"

        print(f"正在截取音频最后 {self.tail_minutes} 分钟并提交给模型定位...")
        location_result = self.asr.recognize(system_prompt, prompt, temp_audio_path)

        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

        return location_result


class AudioProcessingPipeline:
    """端到端音频处理流水线：定位 -> 转录 -> 书面化"""

    def __init__(self, model_name, target_minutes=600, chunk_minutes=20, overlap_minutes=10, max_workers=8):
        self.model_name = model_name
        self.target_minutes = target_minutes
        self.chunk_minutes = chunk_minutes
        self.overlap_minutes = overlap_minutes
        self.max_workers = max_workers

    def _get_audio_duration(self, audio_path):
        audio = AudioSegment.from_mp3(audio_path)
        duration_seconds = int(audio.duration_seconds)
        hours, remainder = divmod(duration_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def run(self, audio_path, fuzzy_text_path, asr_sys_prompt, text_sys_prompt):
        audio_path_obj = Path(audio_path)
        base_name = audio_path_obj.stem
        parent_dir = audio_path_obj.parent

        # 定义所有输出文件的路径
        located_text_filename = parent_dir / f"{base_name}_原文定位.md"
        transcript_filename = parent_dir / f"{base_name}_逐字稿.md"
        final_output_filename = parent_dir / f"{base_name}_书面稿.md"

        # ==========================================
        # 步骤 1: 原文精准定位
        # ==========================================
        print("\n[1/3] 开始进行原文精准定位...")
        locator = AudioTailTextLocator(model_name=self.model_name, tail_minutes=self.target_minutes)
        localized_text = locator.process(audio_path, fuzzy_text_path)

        with open(located_text_filename, "w", encoding="utf-8") as f:
            f.write(localized_text)
        print(f"原文定位完成！已保存至：{located_text_filename}")

        # ==========================================
        # 步骤 2: 音频切分与逐字稿生成
        # ==========================================
        print("\n[2/3] 开始进行音频切分与逐字稿生成...")
        asr_prompt = f"\n    【原典参考】\n    {localized_text}\n    "

        transcriber = BatchAudioTranscriber(
            model_name=self.model_name,
            chunk_minutes=self.chunk_minutes,
            overlap_seconds=self.overlap_minutes * 60,
            temperature=0.1,
            top_p=0.95,
            max_workers=self.max_workers
        )

        transcript_text = transcriber.process_full_audio(audio_path, asr_sys_prompt, asr_prompt)

        with open(transcript_filename, "w", encoding="utf-8") as f:
            f.write(transcript_text)
        print(f"逐字稿已保存至: {transcript_filename}")

        # ==========================================
        # 步骤 3: 整理为书面稿
        # ==========================================
        print("\n[3/3] 正在将逐字稿整理为书面稿，请耐心等待...")
        text_bot = GeminiText(model_name=self.model_name, temperature=0.3)
        user_prompt = f"\n【原典参考】\n{localized_text}\n\n【总逐字稿】\n{transcript_text}\n"

        written_text = text_bot.generate(text_sys_prompt, user_prompt)

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
    model_name = "gemini-3-pro-preview"
    audio_path = "摩诃止观-久仁法师/摩诃止观012/摩诃止观012.mp3"
    fuzzy_text_path = "摩诃止观-久仁法师/摩诃止观012/012原文模糊范围.txt"

    # ---------------- 系统提示词 ----------------
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
        model_name=model_name,
        target_minutes=600,  # 用于原文定位截取的时长，可以取很大的数超过音频时长
        chunk_minutes=10,  # 音频切分时长
        overlap_minutes=5,  # 音频重叠时长
        max_workers=12  # 并发数
    )

    pipeline.run(
        audio_path=audio_path,
        fuzzy_text_path=fuzzy_text_path,
        asr_sys_prompt=asr_system_prompt,
        text_sys_prompt=text_system_prompt
    )