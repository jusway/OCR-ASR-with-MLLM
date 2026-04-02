import os
import time
import json
import uuid
import httpx

# Google SDK
from google import genai
from google.genai import types

# OpenAI SDK (用于兼容 MiMo 等服务)
import openai
from openai import OpenAI

from core.utils import OSSAudioUploader, Color
from core.asr_engine import BaseASR


class GeminiOmni(BaseASR):
    """
    使用 Google Gemini 模型进行 ASR。
    """

    def __init__(self, model_name: str, temperature: float = 0.5, top_p: float = 0.95):
        self.client = genai.Client(http_options={'timeout': None})
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p

    def recognize(self, system_prompt: str, prompt: str, audio_file_path: str) -> str:
        print(f"{Color.DARK_PURPLE}[Gemini] 正在处理音频文件: {audio_file_path}")
        
        # 统一进行压缩，压缩后的临时文件自带纯英文路径，顺便解决了 httpx 的中文路径报错问题
        processed_audio_path, is_temp = self._compress_audio(audio_file_path)
        audio_file = None
            
        try:
            print(f"{Color.DARK_PURPLE}[Gemini] 正在上传压缩后的音频文件...")
            audio_file = self.client.files.upload(file=processed_audio_path)

            while audio_file.state.name == "PROCESSING":
                time.sleep(2)
                audio_file = self.client.files.get(name=audio_file.name)

            if audio_file.state.name == "FAILED":
                raise RuntimeError(f"{Color.RED}[Gemini] 云端音频文件处理失败: {audio_file.name}")

            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=self.temperature,
                top_p=self.top_p,
                max_output_tokens=65536,
            )

            max_retries = 5
            for attempt in range(max_retries):
                try:
                    print(f"{Color.DARK_PURPLE}[Gemini] 正在提交给模型处理 (尝试 {attempt + 1}/{max_retries})...")
                    print(f"{Color.DARK_PURPLE}[Gemini] ⏳ 提示：长音频（如1小时）需要2-5分钟的深度理解时间，请耐心等待模型思考，不要关闭程序...")
                    
                    start_wait_time = time.time()
                    response_stream = self.client.models.generate_content_stream(
                        model=self.model_name,
                        contents=[audio_file, prompt],
                        config=config
                    )

                    full_text = ""
                    first_chunk = True
                    
                    for chunk in response_stream:
                        if first_chunk:
                            wait_duration = time.time() - start_wait_time
                            print(f"\n{Color.GREEN}[Gemini] 💡 模型思考完毕！耗时: {wait_duration:.1f} 秒。开始输出:")
                            print(f"{Color.GREEN}[Gemini 输出]: ", end="", flush=True)
                            first_chunk = False
                            
                        text = chunk.text
                        print(text, end="", flush=True)
                        full_text += text
                    print() # 换行

                    return full_text
                except Exception as e:
                    # 捕获网络断开、超时等异常
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * 5
                        print(f"\n{Color.RED}[Gemini] 发生网络或API异常 ({type(e).__name__})，等待 {wait_time} 秒后重试...")
                        time.sleep(wait_time)
                    else:
                        print(f"\n{Color.RED}[Gemini] 重试次数已达上限，放弃请求。")
                        raise e
        finally:
            # 清理本地临时压缩文件
            if is_temp and os.path.exists(processed_audio_path):
                os.remove(processed_audio_path)
            # 确保无论成功还是失败，都清理云端的音频文件，防止泄漏和占用空间
            if audio_file:
                try:
                    self.client.files.delete(name=audio_file.name)
                except Exception:
                    pass


class MiMoOmni(BaseASR):
    """
    使用 Xiaomi MiMo (如 mimo-v2-omni) 模型进行 ASR。
    """

    def __init__(self, model_name: str = "mimo-v2-omni", temperature: float = 0.5, top_p: float = 0.95):
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
        print(f"{Color.DARK_PURPLE}[MiMo] 正在处理音频文件: {audio_file_path}")

        uploader = OSSAudioUploader()
        original_filename = os.path.basename(audio_file_path)
        base_name, _ = os.path.splitext(original_filename)
        # 添加 UUID 确保每次上传的文件名唯一，避免复用损坏的旧文件
        target_filename = f"{base_name}_{uuid.uuid4().hex[:8]}.mp3"

        is_temp = False
        processed_audio_path = None

        try:
            # 强制进行压缩并上传新文件
            processed_audio_path, is_temp = self._compress_audio(audio_file_path)
            signed_url, object_key = uploader.upload(processed_audio_path, filename=target_filename)

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

            print(f"{Color.DARK_PURPLE}[MiMo] 正在提交给模型处理...")
            print(f"{Color.DARK_PURPLE}[MiMo] ⏳ 提示：长音频（如1小时）需要较长时间的深度理解，请耐心等待模型思考，不要关闭程序...")

            max_retries = 6
            for attempt in range(max_retries):
                try:
                    start_wait_time = time.time()
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
                    first_chunk = True
                    
                    for chunk in response_stream:
                        # 流式处理，获取增量文本
                        if chunk.choices and chunk.choices[0].delta.content:
                            if first_chunk:
                                wait_duration = time.time() - start_wait_time
                                print(f"\n{Color.GREEN}[MiMo] 💡 模型思考完毕！耗时: {wait_duration:.1f} 秒。开始输出:")
                                print(f"{Color.GREEN}[MiMo 输出]: ", end="", flush=True)
                                first_chunk = False
                                
                            text_chunk = chunk.choices[0].delta.content
                            print(text_chunk, end="", flush=True)
                            full_text += text_chunk
                            
                            # === 新增：防无限复读打断机制 ===
                            # 如果最后 30 个字符完全一样（例如连续 30 个"五"），强制抛出异常触发重试
                            if len(full_text) >= 30 and full_text[-30:] == full_text[-1] * 30:
                                print(f"\n{Color.RED}[MiMo] ⚠️ 检测到单字无限复读幻觉，强制中断并重试...")
                                raise RuntimeError("检测到无限复读幻觉")
                    print() # 换行

                    return full_text
                
                except openai.RateLimitError as e:
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * 5  # 指数退避: 5s, 10s, 20s, 40s...
                        print(f"\n{Color.RED}[MiMo] 触发并发限制 (429 Too many requests)，等待 {wait_time} 秒后重试 (第 {attempt + 1}/{max_retries} 次)...")
                        time.sleep(wait_time)
                    else:
                        print(f"\n{Color.RED}[MiMo] 重试次数已达上限，放弃请求。")
                        raise e
                except openai.BadRequestError as e:
                    error_msg = str(e)
                    # 捕获内容风控拦截 (Moderation Block / 421)
                    if '421' in error_msg or 'content_filter' in error_msg or 'Moderation Block' in error_msg:
                        print(f"\n{Color.RED}[MiMo] ⚠️ 警告: 该片段触发了 API 的安全审核策略 (Moderation Block)，已被拦截。将跳过此片段。")
                        return "\n[⚠️ 此片段内容被 API 安全策略拦截，无法转录]\n"
                    else:
                        # 其他的 BadRequestError 正常抛出
                        raise e
                except RuntimeError as e:
                    if "检测到无限复读幻觉" in str(e):
                        if attempt < max_retries - 1:
                            wait_time = (2 ** attempt) * 5
                            print(f"\n{Color.RED}[MiMo] 触发复读机幻觉，等待 {wait_time} 秒后重试 (第 {attempt + 1}/{max_retries} 次)...")
                            time.sleep(wait_time)
                        else:
                            print(f"\n{Color.RED}[MiMo] 重试次数已达上限，放弃请求。")
                            raise e
                    else:
                        raise e

        finally:
            # 清理可能生成的临时压缩文件
            if is_temp and processed_audio_path and os.path.exists(processed_audio_path):
                os.remove(processed_audio_path)


class Qwen3OmniFlash(BaseASR):
    """
    使用 Qwen3-Omni-Flash-2025-12-01 模型进行 ASR (基于阿里云原生多模态 API)。
    """

    def __init__(self, model_name: str = "qwen3-omni-flash-2025-12-01", temperature: float = 0.5, top_p: float = 0.95):
        self.api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("未检测到 DASHSCOPE_API_KEY 环境变量，请先设置阿里云 API Key。")

        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.api_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

    def recognize(self, system_prompt: str, prompt: str, audio_file_path: str) -> str:
        print(f"{Color.DARK_PURPLE}[Qwen3OmniFlash] 正在处理音频文件: {audio_file_path}")

        uploader = OSSAudioUploader()
        original_filename = os.path.basename(audio_file_path)
        base_name, _ = os.path.splitext(original_filename)
        target_filename = f"{base_name}_{uuid.uuid4().hex[:8]}.mp3"

        is_temp = False
        processed_audio_path = None

        try:
            # 强制进行压缩并上传新文件
            processed_audio_path, is_temp = self._compress_audio(audio_file_path)
            signed_url, object_key = uploader.upload(processed_audio_path, filename=target_filename)

            # 构建阿里云百炼原生多模态 API 的请求体
            payload = {
                "model": self.model_name,
                "input": {
                    "messages": [
                        {
                            "role": "system",
                            "content": [{"text": system_prompt}]
                        },
                        {
                            "role": "user",
                            "content": [
                                {"audio": signed_url},
                                {"text": prompt}
                            ]
                        }
                    ]
                },
                "parameters": {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "incremental_output": True  # 开启增量流式输出
                }
            }

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "X-DashScope-SSE": "enable",  # 开启 SSE 流式响应
                "X-DashScope-OssResourceResolve": "enable"  # 允许解析 OSS 等临时链接
            }

            print(f"{Color.DARK_PURPLE}[Qwen3OmniFlash] 正在提交给模型处理...")
            print(f"{Color.DARK_PURPLE}[Qwen3OmniFlash] ⏳ 提示：长音频需要较长时间的深度理解，请耐心等待模型思考，不要关闭程序...")

            max_retries = 6
            for attempt in range(max_retries):
                try:
                    full_text = ""
                    first_chunk = True
                    start_wait_time = time.time()
                    
                    with httpx.Client(timeout=None) as client:
                        with client.stream("POST", self.api_url, headers=headers, json=payload) as response:
                            if response.status_code != 200:
                                error_text = response.read().decode('utf-8')
                                if "Throttling" in error_text or "TooManyRequests" in error_text:
                                    raise Exception(f"RateLimit: {error_text}")
                                elif "DataInspectionFailed" in error_text:
                                    print(f"\n{Color.RED}[Qwen3OmniFlash] ⚠️ 警告: 该片段触发了 API 的安全审核策略，已被拦截。将跳过此片段。")
                                    return "\n[⚠️ 此片段内容被 API 安全策略拦截，无法转录]\n"
                                else:
                                    if response.status_code == 403:
                                        print(f"\n{Color.RED}[Qwen3OmniFlash] ⚠️ 403 AccessDenied: 请确保您已在阿里云百炼控制台开通并申请了该模型的使用权限。")
                                    raise RuntimeError(f"DashScope API Error ({response.status_code}): {error_text}")

                            for line in response.iter_lines():
                                if line.startswith("data:"):
                                    data_str = line[5:].strip()
                                    if not data_str:
                                        continue
                                    try:
                                        data = json.loads(data_str)
                                        choices = data.get("output", {}).get("choices", [])
                                        if choices:
                                            content = choices[0].get("message", {}).get("content", [])
                                            if isinstance(content, list):
                                                for item in content:
                                                    if "text" in item:
                                                        if first_chunk:
                                                            wait_duration = time.time() - start_wait_time
                                                            print(f"\n{Color.GREEN}[Qwen3OmniFlash] 💡 模型思考完毕！耗时: {wait_duration:.1f} 秒。开始输出:")
                                                            print(f"{Color.GREEN}[Qwen3OmniFlash 输出]: ", end="", flush=True)
                                                            first_chunk = False
                                                        print(item["text"], end="", flush=True)
                                                        full_text += item["text"]
                                            elif isinstance(content, str):
                                                if first_chunk:
                                                    wait_duration = time.time() - start_wait_time
                                                    print(f"\n{Color.GREEN}[Qwen3OmniFlash] 💡 模型思考完毕！耗时: {wait_duration:.1f} 秒。开始输出:")
                                                    print(f"{Color.GREEN}[Qwen3OmniFlash 输出]: ", end="", flush=True)
                                                    first_chunk = False
                                                print(content, end="", flush=True)
                                                full_text += content
                                    except json.JSONDecodeError:
                                        pass
                    print() # 换行
                    return full_text
                
                except Exception as e:
                    error_msg = str(e)
                    if "RateLimit" in error_msg or isinstance(e, httpx.RequestError):
                        if attempt < max_retries - 1:
                            wait_time = (2 ** attempt) * 5
                            print(f"\n{Color.RED}[Qwen3OmniFlash] 触发并发限制或网络异常，等待 {wait_time} 秒后重试 (第 {attempt + 1}/{max_retries} 次)...")
                            time.sleep(wait_time)
                        else:
                            print(f"\n{Color.RED}[Qwen3OmniFlash] 重试次数已达上限，放弃请求。")
                            raise e
                    else:
                        raise e

        finally:
            # 清理可能生成的临时压缩文件
            if is_temp and processed_audio_path and os.path.exists(processed_audio_path):
                os.remove(processed_audio_path)


class QwenOmni(BaseASR):
    """
    使用 Qwen 3.5 Omni 模型进行 ASR。
    由于阿里云百炼的 OpenAI 兼容接口暂不支持 audio_url，这里改用原生多模态 API。
    """

    def __init__(self, model_name: str = "qwen3.5-omni-plus", temperature: float = 0.5, top_p: float = 0.95):
        self.api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("未检测到 DASHSCOPE_API_KEY 环境变量，请先设置阿里云 API Key。")

        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.api_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

    def recognize(self, system_prompt: str, prompt: str, audio_file_path: str) -> str:
        print(f"{Color.DARK_PURPLE}[Qwen] 正在处理音频文件: {audio_file_path}")

        uploader = OSSAudioUploader()
        original_filename = os.path.basename(audio_file_path)
        base_name, _ = os.path.splitext(original_filename)
        # 添加 UUID 确保每次上传的文件名唯一
        target_filename = f"{base_name}_{uuid.uuid4().hex[:8]}.mp3"

        is_temp = False
        processed_audio_path = None

        try:
            # 强制进行压缩并上传新文件
            processed_audio_path, is_temp = self._compress_audio(audio_file_path)
            signed_url, object_key = uploader.upload(processed_audio_path, filename=target_filename)

            # 构建阿里云百炼原生多模态 API 的请求体
            payload = {
                "model": self.model_name,
                "input": {
                    "messages": [
                        {
                            "role": "system",
                            "content": [{"text": system_prompt}]
                        },
                        {
                            "role": "user",
                            "content": [
                                {"audio": signed_url},
                                {"text": prompt}
                            ]
                        }
                    ]
                },
                "parameters": {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "incremental_output": True  # 开启增量流式输出
                }
            }

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "X-DashScope-SSE": "enable",  # 开启 SSE 流式响应
                "X-DashScope-OssResourceResolve": "enable"  # 允许解析 OSS 等临时链接
            }

            print(f"{Color.DARK_PURPLE}[Qwen] 正在提交给模型处理...")
            print(f"{Color.DARK_PURPLE}[Qwen] ⏳ 提示：长音频（如1小时）需要较长时间的深度理解，请耐心等待模型思考，不要关闭程序...")

            max_retries = 6
            for attempt in range(max_retries):
                try:
                    full_text = ""
                    first_chunk = True
                    start_wait_time = time.time()
                    
                    with httpx.Client(timeout=None) as client:
                        with client.stream("POST", self.api_url, headers=headers, json=payload) as response:
                            if response.status_code != 200:
                                error_text = response.read().decode('utf-8')
                                if "Throttling" in error_text or "TooManyRequests" in error_text:
                                    raise Exception(f"RateLimit: {error_text}")
                                elif "DataInspectionFailed" in error_text:
                                    print(f"\n{Color.RED}[Qwen] ⚠️ 警告: 该片段触发了 API 的安全审核策略，已被拦截。将跳过此片段。")
                                    return "\n[⚠️ 此片段内容被 API 安全策略拦截，无法转录]\n"
                                else:
                                    if response.status_code == 403:
                                        print(f"\n{Color.RED}[Qwen] ⚠️ 403 AccessDenied: 请确保您已在阿里云百炼控制台开通并申请了该模型的使用权限，或者检查 OSS 链接是否可访问。")
                                    raise RuntimeError(f"DashScope API Error ({response.status_code}): {error_text}")

                            for line in response.iter_lines():
                                if line.startswith("data:"):
                                    data_str = line[5:].strip()
                                    if not data_str:
                                        continue
                                    try:
                                        data = json.loads(data_str)
                                        choices = data.get("output", {}).get("choices", [])
                                        if choices:
                                            content = choices[0].get("message", {}).get("content", [])
                                            if isinstance(content, list):
                                                for item in content:
                                                    if "text" in item:
                                                        if first_chunk:
                                                            wait_duration = time.time() - start_wait_time
                                                            print(f"\n{Color.GREEN}[Qwen] 💡 模型思考完毕！耗时: {wait_duration:.1f} 秒。开始输出:")
                                                            print(f"{Color.GREEN}[Qwen 输出]: ", end="", flush=True)
                                                            first_chunk = False
                                                        print(item["text"], end="", flush=True)
                                                        full_text += item["text"]
                                            elif isinstance(content, str):
                                                if first_chunk:
                                                    wait_duration = time.time() - start_wait_time
                                                    print(f"\n{Color.GREEN}[Qwen] 💡 模型思考完毕！耗时: {wait_duration:.1f} 秒。开始输出:")
                                                    print(f"{Color.GREEN}[Qwen 输出]: ", end="", flush=True)
                                                    first_chunk = False
                                                print(content, end="", flush=True)
                                                full_text += content
                                    except json.JSONDecodeError:
                                        pass
                    print() # 换行
                    return full_text
                
                except Exception as e:
                    error_msg = str(e)
                    if "RateLimit" in error_msg or isinstance(e, httpx.RequestError):
                        if attempt < max_retries - 1:
                            wait_time = (2 ** attempt) * 5
                            print(f"\n{Color.RED}[Qwen] 触发并发限制或网络异常，等待 {wait_time} 秒后重试 (第 {attempt + 1}/{max_retries} 次)...")
                            time.sleep(wait_time)
                        else:
                            print(f"\n{Color.RED}[Qwen] 重试次数已达上限，放弃请求。")
                            raise e
                    else:
                        # 其他不可恢复的错误直接抛出
                        raise e

        finally:
            # 清理可能生成的临时压缩文件
            if is_temp and processed_audio_path and os.path.exists(processed_audio_path):
                os.remove(processed_audio_path)
