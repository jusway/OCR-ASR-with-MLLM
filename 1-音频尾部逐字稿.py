import os
import uuid
from pathlib import Path
from pydub import AudioSegment
from core.gemini_asr import GeminiASR


class AudioTailTranscriber:
    def __init__(self,model_name, tail_minutes: int = 5, temperature: float = 0.1, top_p: float = 0.95):
        self.asr = GeminiASR(temperature=temperature, top_p=top_p,model_name=model_name)
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

    def process(self, audio_path: str) -> str:
        temp_audio_path = self._extract_tail(audio_path)

        system_prompt = (
            "你是一位资深的佛法音频转录义工。\n"
            "将用户提供的音频准确转录为逐字稿。\n"
            "保留所有的停顿、口语化表达和重复等所有内容，对音频语音完全忠实。\n"
            "使用简体字输出。遇到原典引用的时候请使用『』符号。\n"
            "只输出转录内容，不说多余的话。"
        )

        prompt = "请转录这段音频。"

        print(f"正在截取音频最后{self.tail_minutes}分钟并提交给模型处理...")
        transcription = self.asr.recognize(system_prompt, prompt, temp_audio_path)

        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

        return transcription


if __name__ == "__main__":
    audio_path = "摩诃止观-久仁法师/摩诃止观015/摩诃止观015.mp3"
    target_minutes = 20
    model_name="gemini-3.1-pro-preview"

    transcriber = AudioTailTranscriber(model_name=model_name,tail_minutes=target_minutes)
    final_text = transcriber.process(audio_path)

    audio_p = Path(audio_path)
    output_filename = audio_p.parent / f"{audio_p.stem}_结尾{target_minutes}分钟逐字稿.md"

    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(final_text)

    print(f"处理完成！文件已保存至：{output_filename}")