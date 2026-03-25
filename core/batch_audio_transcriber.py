import os
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from core.audio_chunker import AudioChunker
from core.asr import GeminiASR


class BatchAudioTranscriber:
    def __init__(self, model_name, chunk_minutes=3, overlap_seconds=10, temperature=0.1, top_p=0.95, max_workers=3):
        self.chunk_minutes = chunk_minutes
        self.overlap_seconds = overlap_seconds
        self.chunker = AudioChunker(chunk_minutes=chunk_minutes, overlap_seconds=overlap_seconds)
        self.asr = GeminiASR(model_name=model_name, temperature=temperature, top_p=top_p)
        self.max_workers = max_workers
        self.lock = threading.Lock()
        self.completed_count = 0

    def _process_single_chunk(self, chunk_file, chunk_index, system_prompt, prompt, total_chunks):
        chunk_path = Path(chunk_file)
        md_file = chunk_path.with_suffix('.md')

        if md_file.exists():
            with self.lock:
                self.completed_count += 1
                current_count = self.completed_count
            print(f"--- 已完成 {current_count}/{total_chunks} 个片段 (直接读取缓存: {md_file.name}) ---")
            with open(md_file, "r", encoding="utf-8") as f:
                return f.read(), chunk_file

        # 调用 recognize (GeminiASR 内部已使用 self.model_name)
        chunk_text = self.asr.recognize(system_prompt, prompt, chunk_file)

        header = f"# 片段 {chunk_index}/{total_chunks}\n\n"
        final_content = header + chunk_text

        with open(md_file, "w", encoding="utf-8") as f:
            f.write(final_content)

        with self.lock:
            self.completed_count += 1
            current_count = self.completed_count
        print(f"--- 已完成 {current_count}/{total_chunks} 个片段 ---")

        return final_content, chunk_file

    def process_full_audio(self, audio_path, system_prompt, prompt):
        self.completed_count = 0
        audio_path_obj = Path(audio_path)
        base_name = audio_path_obj.stem
        parent_dir = audio_path_obj.parent

        chunk_output_dir = parent_dir / f"{base_name}_chunks_{self.chunk_minutes}m_{self.overlap_seconds}s"
        os.makedirs(chunk_output_dir, exist_ok=True)

        chunk_files = self.chunker.process(audio_path, output_dir=str(chunk_output_dir))

        full_transcription = ""
        total_chunks = len(chunk_files)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交任务时，顺序保持与切分顺序一致
            futures = []
            for i, chunk_file in enumerate(chunk_files):
                future = executor.submit(self._process_single_chunk, chunk_file, i , system_prompt, prompt,
                                         total_chunks)
                futures.append(future)

            for future in futures:
                chunk_text, chunk_file = future.result()
                full_transcription += chunk_text + "\n\n---\n\n"

        return full_transcription


if __name__ == "__main__":
    system_prompt = (
        # 角色
        "你是一位资深的佛法音频转录工作的出家比丘大师。"
        # 任务
        "将用户提供的音频准确转录为逐字稿。"
        # 输出规范
        "保留所有的停顿、口语化表达和重复等所有内容，对音频语音完全忠实。"
        "严格按照用户提供的【原典参考】校对专有名词。"
        "引用原典的时候使用『』符号。"
        "只输出转录内容，不说多余的话。"
    )

    prompt_path = r"摩诃止观-久仁法师/摩诃止观016/016原文.txt"
    audio_path = "摩诃止观-久仁法师/摩诃止观016/摩诃止观016.mp3"
    chunk_minutes = 20
    overlap_minutes=10
    max_workers = 8
    model_name = "gemini-3.1-pro-preview"


    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_text = f.read()
    prompt = f"""
    【原典参考】
    {prompt_text}
    """

    audio_path_obj = Path(audio_path)
    output_filename = audio_path_obj.parent / f"{audio_path_obj.stem}_逐字稿.md"

    pipeline = BatchAudioTranscriber(
        model_name=model_name,  # 传入模型名
        chunk_minutes=chunk_minutes,
        overlap_seconds=overlap_minutes*60,
        temperature=0.1,
        top_p=0.95,
        max_workers=max_workers
    )

    print(f"开始进行音频切分与批量处理 (使用模型: {model_name})...")
    final_text = pipeline.process_full_audio(audio_path, system_prompt, prompt)
    print("\n所有片段处理完成！")

    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(final_text)