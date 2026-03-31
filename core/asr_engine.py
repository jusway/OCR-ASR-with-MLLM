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
import openai
from openai import OpenAI
import httpx

from core.utils import AudioCompressor, OSSAudioUploader


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

        try:
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

            max_retries = 5
            for attempt in range(max_retries):
                try:
                    print(f"[Gemini] 正在提交给模型处理 (尝试 {attempt + 1}/{max_retries})...")
                    response_stream = self.client.models.generate_content_stream(
                        model=self.model_name,
                        contents=[audio_file, prompt],
                        config=config
                    )

                    full_text = ""
                    for chunk in response_stream:
                        text = chunk.text
                        full_text += text

                    return full_text
                except Exception as e:
                    # 捕获网络断开、超时等异常
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * 5
                        print(f"\n[Gemini] 发生网络或API异常 ({type(e).__name__})，等待 {wait_time} 秒后重试...")
                        time.sleep(wait_time)
                    else:
                        print(f"\n[Gemini] 重试次数已达上限，放弃请求。")
                        raise e
        finally:
            # 确保无论成功还是失败，都清理云端的音频文件，防止泄漏和占用空间
            try:
                self.client.files.delete(name=audio_file.name)
            except Exception:
                pass


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
        print(f"[MiMo] 正在处理音频文件: {audio_file_path}")

        processed_audio_path, is_temp = AudioCompressor.compress_if_needed(audio_file_path, max_size_mb=50)

        try:
            # 使用 OSS 上传获取 URL
            uploader = OSSAudioUploader()
            signed_url, _ = uploader.upload(processed_audio_path)

            # 构建 OpenAI 兼容格式的消息，使用 audio_url
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "audio_url",
                            "audio_url": {
                                "url": signed_url
                            }
                        }
                    ]
                }
            ]

            print("[MiMo] 正在提交给模型处理...")

            max_retries = 6
            for attempt in range(max_retries):
                try:
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
                            full_text += text_chunk

                    return full_text
                
                except openai.RateLimitError as e:
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * 5  # 指数退避: 5s, 10s, 20s, 40s...
                        print(f"\n[MiMo] 触发并发限制 (429 Too many requests)，等待 {wait_time} 秒后重试 (第 {attempt + 1}/{max_retries} 次)...")
                        time.sleep(wait_time)
                    else:
                        print(f"\n[MiMo] 重试次数已达上限，放弃请求。")
                        raise e
                except openai.BadRequestError as e:
                    error_msg = str(e)
                    # 捕获内容风控拦截 (Moderation Block / 421)
                    if '421' in error_msg or 'content_filter' in error_msg or 'Moderation Block' in error_msg:
                        print(f"\n[MiMo] ⚠️ 警告: 该片段触发了 API 的安全审核策略 (Moderation Block)，已被拦截。将跳过此片段。")
                        return "\n[⚠️ 此片段内容被 API 安全策略拦截，无法转录]\n"
                    else:
                        # 其他的 BadRequestError 正常抛出
                        raise e

        finally:
            # 清理可能生成的临时压缩文件
            if is_temp and os.path.exists(processed_audio_path):
                os.remove(processed_audio_path)


class QwenASR(BaseASR):
    """
    使用 Qwen 3.5 Omni 模型进行 ASR。
    """

    def __init__(self, model_name: str = "qwen3.5-omni-plus", temperature: float = 0.1, top_p: float = 0.95):
        # 请确保在使用前在环境变量中设置了 DASHSCOPE_API_KEY
        self.api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("未检测到 DASHSCOPE_API_KEY 环境变量，请先设置阿里云 API Key。")

        # 使用 openai 客户端指向阿里云百炼兼容接口
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p

    def recognize(self, system_prompt: str, prompt: str, audio_file_path: str) -> str:
        print(f"[Qwen] 正在处理音频文件: {audio_file_path}")

        processed_audio_path, is_temp = AudioCompressor.compress_if_needed(audio_file_path, max_size_mb=50)

        try:
            # 使用 OSS 上传获取 URL
            uploader = OSSAudioUploader()
            signed_url, _ = uploader.upload(processed_audio_path)

            # 构建 OpenAI 兼容格式的消息，使用 audio_url
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "audio_url",
                            "audio_url": {
                                "url": signed_url
                            }
                        }
                    ]
                }
            ]

            print("[Qwen] 正在提交给模型处理...")

            max_retries = 6
            for attempt in range(max_retries):
                try:
                    response_stream = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        modalities=["text"], # 仅需要文本输出
                        stream=True,         # 官方文档要求必须为 True
                        stream_options={"include_usage": True}
                    )

                    full_text = ""
                    for chunk in response_stream:
                        # 流式处理，获取增量文本
                        if chunk.choices and chunk.choices[0].delta.content:
                            text_chunk = chunk.choices[0].delta.content
                            full_text += text_chunk

                    return full_text
                
                except openai.RateLimitError as e:
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * 5  # 指数退避: 5s, 10s, 20s, 40s...
                        print(f"\n[Qwen] 触发并发限制 (429 Too many requests)，等待 {wait_time} 秒后重试 (第 {attempt + 1}/{max_retries} 次)...")
                        time.sleep(wait_time)
                    else:
                        print(f"\n[Qwen] 重试次数已达上限，放弃请求。")
                        raise e
                except openai.BadRequestError as e:
                    error_msg = str(e)
                    # 捕获内容风控拦截
                    if '421' in error_msg or 'content_filter' in error_msg or 'Moderation Block' in error_msg:
                        print(f"\n[Qwen] ⚠️ 警告: 该片段触发了 API 的安全审核策略，已被拦截。将跳过此片段。")
                        return "\n[⚠️ 此片段内容被 API 安全策略拦截，无法转录]\n"
                    else:
                        # 其他的 BadRequestError 正常抛出
                        raise e

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

    prompt_path = r"../摩诃止观-久仁法师/摩诃止观001/001原文模糊范围.txt"  # 记载原文内容
    audio_path = "../摩诃止观-久仁法师/摩诃止观001/摩诃止观001.mp3"

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
    # asr_engine = MiMoASR(model_name="mimo-v2-omni", temperature=0.1, top_p=0.95)

    # 或者使用 Qwen:
    # 请确保在终端执行前：export DASHSCOPE_API_KEY="你的key"
    print("正在初始化 QwenASR 引擎...")
    asr_engine = QwenASR(model_name="qwen3.5-omni-plus", temperature=0.1, top_p=0.95)

    print("--- 任务开始 ---")
    full_transcription = asr_engine.recognize(system_prompt, prompt, audio_path)
    print("--- 处理完成 ---")

    output_filename = f"{Path(audio_path).stem}_{type(asr_engine).__name__}_逐字稿.md"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(full_transcription)

    print(f"✅ 文件已保存为: {output_filename}")
