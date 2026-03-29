import os
import time
import base64
import tempfile
import shutil
from abc import ABC, abstractmethod
from pathlib import Path

# Google SDK
from google import genai
from google.genai import types

# OpenAI SDK (用于兼容 MiMo 等服务)
from openai import OpenAI


class AudioCompressor:
    """
    音频压缩工具类，用于处理过大的音频文件。
    """
    @staticmethod
    def compress_if_needed(audio_path: str, max_size_mb: int = 50) -> tuple[str, bool]:
        """
        检查音频文件大小，如果超过指定阈值则进行压缩。
        返回: (处理后的音频路径, 是否生成了临时文件)
        """
        file_size = os.path.getsize(audio_path)
        max_size_bytes = max_size_mb * 1024 * 1024

        if file_size <= max_size_bytes:
            return audio_path, False

        print(f"[AudioCompressor] 音频文件过大 ({file_size / 1024 / 1024:.2f} MB)，超过 {max_size_mb}MB 阈值，正在压缩...")
        try:
            from pydub import AudioSegment
        except ImportError:
            raise RuntimeError("需要安装 pydub 来压缩过大的音频文件。请运行 `pip install pydub`。")
        
        audio = AudioSegment.from_file(audio_path)
        
        # 压缩原理说明：
        # 1. set_channels(1): 转换为单声道，去除立体声的冗余数据。
        # 2. set_frame_rate(16000): 降低采样率到 16kHz，保留人声频段，丢弃无用的高频细节。
        audio = audio.set_channels(1).set_frame_rate(16000)
        
        fd, temp_compressed_path = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)
        
        # 3. bitrate="32k": 使用 MP3 有损压缩，并限制极低的比特率（32kbps），在保证语音可辨识的前提下极限压缩体积。
        audio.export(temp_compressed_path, format="mp3", bitrate="32k")
        print(f"[AudioCompressor] 压缩完成，新文件大小: {os.path.getsize(temp_compressed_path) / 1024 / 1024:.2f} MB")
        
        return temp_compressed_path, True


class BaseASR(ABC):
    """
    ASR (语音识别) 抽象基类，规范统一的调用接口。
    """

    @abstractmethod
    def recognize(self, system_prompt: str, prompt: str, audio_file_path: str) -> str:
        """
        根据提供的系统提示词、用户提示词和音频文件路径，返回转录文本。
        """
        pass


class GeminiASR(BaseASR):
    """
    使用 Google Gemini 模型进行 ASR。
    """

    def __init__(self, model_name: str, temperature: float = 0.1, top_p: float = 0.95):
        self.client = genai.Client(http_options={'timeout': None})
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p

    def recognize(self, system_prompt: str, prompt: str, audio_file_path: str) -> str:
        print(f"[Gemini] 正在准备上传音频文件: {audio_file_path}")
        
        # 解决中文路径导致 httpx 报 UnicodeEncodeError 的问题
        # 创建一个纯英文的临时文件进行上传
        ext = os.path.splitext(audio_file_path)[1]
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as temp_file:
            temp_path = temp_file.name
            
        try:
            # 将原文件复制到临时纯英文路径
            shutil.copy2(audio_file_path, temp_path)
            print(f"[Gemini] 已创建临时文件以规避中文路径编码问题: {temp_path}")
            
            # 使用临时文件上传
            audio_file = self.client.files.upload(file=temp_path)
        finally:
            # 无论上传成功与否，清理本地临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)

        while audio_file.state.name == "PROCESSING":
            time.sleep(2)
            audio_file = self.client.files.get(name=audio_file.name)

        if audio_file.state.name == "FAILED":
            raise RuntimeError(f"[Gemini] 云端音频文件处理失败: {audio_file.name}")

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=self.temperature,
            top_p=self.top_p,
            max_output_tokens=65536,
        )

        print("[Gemini] 正在提交给模型处理，采用流式输出...")
        response_stream = self.client.models.generate_content_stream(
            model=self.model_name,
            contents=[audio_file, prompt],
            config=config
        )

        full_text = ""
        for chunk in response_stream:
            text = chunk.text
            print(text, end="", flush=True)
            full_text += text

        print()  # 换行
        self.client.files.delete(name=audio_file.name)
        return full_text


class MiMoASR(BaseASR):
    """
    使用 Xiaomi MiMo (如 mimo-v2-omni) 模型进行 ASR。
    """

    def __init__(self, model_name: str = "mimo-v2-omni", temperature: float = 0.1, top_p: float = 0.95):
        # 请确保在使用前在环境变量中设置了 MIMO_API_KEY
        self.api_key = os.environ.get("MIMO_API_KEY")
        if not self.api_key:
            raise ValueError("未检测到 MIMO_API_KEY 环境变量，请先设置小米 API Key。")

        # 使用 openai 客户端指向 MiMo 兼容接口
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.xiaomimimo.com/v1"
        )
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p

    def recognize(self, system_prompt: str, prompt: str, audio_file_path: str) -> str:
        print(f"[MiMo] 正在读取并编码音频文件: {audio_file_path}")

        processed_audio_path, is_temp = AudioCompressor.compress_if_needed(audio_file_path, max_size_mb=50)

        try:
            # 根据扩展名获取格式，常见的为 mp3 或 wav
            audio_ext = os.path.splitext(processed_audio_path)[-1].lower().replace(".", "")
            if audio_ext not in ['mp3', 'wav']:
                print(f"⚠️ 警告: 未知格式 '{audio_ext}'，使用 'wav' 作为默认兼容格式。")
                audio_ext = "wav"

            # 读取音频并编码为 Base64
            with open(processed_audio_path, "rb") as audio_file:
                audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')

            # 构建 OpenAI 兼容格式的消息
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": audio_base64,
                                "format": audio_ext
                            }
                        }
                    ]
                }
            ]

            print("[MiMo] 正在提交给模型处理，采用流式输出...")

            # 官方文档指出 mimo-v2-omni 默认最大 token 是 32768
            response_stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=32768,
                stream=True
            )

            full_text = ""
            for chunk in response_stream:
                # 流式处理，获取增量文本
                if chunk.choices and chunk.choices[0].delta.content:
                    text_chunk = chunk.choices[0].delta.content
                    print(text_chunk, end="", flush=True)
                    full_text += text_chunk

            print()  # 换行
            return full_text
        finally:
            # 清理可能生成的临时压缩文件
            if is_temp and os.path.exists(processed_audio_path):
                os.remove(processed_audio_path)


if __name__ == "__main__":
    system_prompt = (
        # 角色
        "你是顶尖的佛法音频转录专家。"
        # 任务
        "将用户提供的音频准确转录为Markdown格式逐字稿。"
        # 输出规范
        "保留所有的停顿、口语化表达和重复等所有内容，对音频语音完全忠实。"
        "严格按照用户提供的【原典参考】校对专有名词。"
        "使用简体字输出。"
        "原典引用的时候使用繁体字。"
    )

    prompt_path = r"../摩诃止观-久仁法师/摩诃止观012/012原文模糊范围.txt"  # 记载原文内容
    audio_path = "../摩诃止观-久仁法师/摩诃止观012/chunk_001.mp3"

    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_text = f.read()
    except FileNotFoundError:
        # 占位符，避免在找不到文件时中断测试
        prompt_text = "测试文本..."

    prompt = f"""
    【原典参考】
    {prompt_text}
    """

    # --- 灵活切换 ASR 引擎 ---
    # 使用 Gemini:
    # asr_engine = GeminiASR(model_name="gemini-3.1-pro-preview", temperature=0.1, top_p=0.95)

    # 或者使用 MiMo:
    # 请确保在终端执行前：export MIMO_API_KEY="你的key"
    asr_engine = MiMoASR(model_name="mimo-v2-omni", temperature=0.1, top_p=0.95)

    print("--- 任务开始 ---")
    full_transcription = asr_engine.recognize(system_prompt, prompt, audio_path)
    print("--- 处理完成 ---")

    output_filename = f"{Path(audio_path).stem}_{type(asr_engine).__name__}_逐字稿.md"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(full_transcription)

    print(f"✅ 文件已保存为: {output_filename}")
