import os
import re
import time
import json
import uuid
from abc import ABC, abstractmethod
import httpx
import requests

import dashscope
from dashscope.audio.qwen_asr import QwenTranscription

from core.utils import AudioCompressor, OSSAudioUploader, Color


class BaseASR(ABC):
    """
    ASR (语音识别) 抽象基类，规范统一的调用接口。特点是声学对齐。
    """

    def _compress_audio(self, audio_file_path: str) -> tuple[str, bool]:
        """
        内部方法：将音频统一压缩为大模型标准 (16kHz 单声道 mp3)。
        返回: (处理后的音频路径, 是否生成了临时文件)
        """
        return AudioCompressor.compress(audio_file_path)

    @abstractmethod
    def recognize(self, audio_file_path: str) -> str:
        """
        根据提供的音频文件路径，返回转录文本。
        """
        pass


class Qwen3ASRFlashFiletrans(BaseASR):
    """
    使用 Qwen3-ASR-Flash-Filetrans 模型进行异步语音识别 (基于 DashScope SDK)。
    """

    def __init__(self, model_name: str = "qwen3-asr-flash-filetrans"):
        self.api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("未检测到 DASHSCOPE_API_KEY 环境变量，请先设置阿里云 API Key。")

        self.model_name = model_name
        dashscope.api_key = self.api_key

    def recognize(self, audio_file_path: str) -> str:
        print(f"[Qwen3ASRFlash] 正在处理音频文件: {audio_file_path}")

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

            print(f"[Qwen3ASRFlash] 正在提交异步转录任务...")
            
            # 使用 SDK 提交异步任务
            task_response = QwenTranscription.async_call(
                model=self.model_name,
                file_url=signed_url,
                enable_itn=False
            )
            
            if task_response.status_code != 200:
                raise RuntimeError(f"提交任务失败! HTTP code: {task_response.status_code}, Response: {task_response}")

            task_id = task_response.output.task_id
            print(f"[Qwen3ASRFlash] 任务已提交，task_id: {task_id}。正在等待任务完成...")

            # 使用 SDK 的 wait 方法自动轮询等待结果（含重试）
            for attempt in range(5):
                try:
                    task_result = QwenTranscription.wait(task=task_id)
                    break
                except Exception as e:
                    if attempt < 4:
                        print(f"{Color.RED}⚠️ 查询任务状态失败 (尝试 {attempt+1}/5): {repr(e)}")
                        time.sleep(5)
                    else:
                        raise

            if task_result.status_code != 200:
                raise RuntimeError(f"查询任务状态失败! HTTP code: {task_result.status_code}, Response: {task_result}")

            status = task_result.output.task_status

            if status.upper() == "SUCCEEDED":
                print(f"{Color.GREEN}[Qwen3ASRFlash] 任务已完成！正在提取转录结果...")

                # 修复：阿里云返回的是单数的 result 对象，而不是 results 数组
                result_obj = task_result.output.get("result", {})
                if not result_obj:
                    print(f"{Color.RED}[Qwen3ASRFlash] 警告：未找到 result 字段。原始 output: {task_result.output}")
                    return ""

                transcription_url = result_obj.get("transcription_url")
                if transcription_url:
                    # 下载结果 JSON
                    with httpx.Client(timeout=30.0) as client:
                        res_resp = client.get(transcription_url)
                        if res_resp.status_code == 200:
                            res_data = res_resp.json()
                            
                            # 兼容多种 JSON 结构提取文本
                            full_text = ""
                            if "transcripts" in res_data:
                                full_text = "".join([t.get("text", "") for t in res_data["transcripts"]])
                            elif "text" in res_data:
                                full_text = res_data["text"]
                            elif isinstance(res_data, list):
                                full_text = "".join([item.get("text", "") for item in res_data if isinstance(item, dict)])
                                
                            if not full_text:
                                print(f"{Color.RED}[Qwen3ASRFlash] 警告：未能提取到文本，原始结果：\n{json.dumps(res_data, ensure_ascii=False, indent=2)}")
                            else:
                                print(f"{Color.GREEN}[Qwen3ASRFlash 输出]:\n{full_text}")
                            return full_text
                        else:
                            raise RuntimeError(f"获取转录结果文件失败: {res_resp.status_code}")
                else:
                    # 如果直接返回了 text
                    text = result_obj.get("text", "")
                    print(f"{Color.GREEN}[Qwen3ASRFlash 输出]:\n{text}")
                    return text
                    
            elif status.upper() == "FAILED":
                raise RuntimeError(f"任务执行失败: {task_result}")
            elif status.upper() == "UNKNOWN":
                raise RuntimeError(f"任务状态未知: {task_result}")
            else:
                raise RuntimeError(f"任务状态异常: {task_result}")

        finally:
            # 清理可能生成的临时压缩文件
            if is_temp and processed_audio_path and os.path.exists(processed_audio_path):
                os.remove(processed_audio_path)


class MiMoLocalASR(BaseASR):
    """
    使用本地运行的 MiMo-V2.5-ASR 模型进行语音识别。

    调用本地 HTTP 服务，默认地址 http://127.0.0.1:8000
    支持语言提示标签 <chinese> / <english>。
    自动将 Windows 绝对路径转换为 WSL 路径（如 D:\path → /mnt/d/path）。
    """

    @staticmethod
    def _to_wsl_path(windows_path: str) -> str:
        """将 Windows 绝对路径转换为 WSL 路径"""
        if not windows_path:
            return windows_path
        path = windows_path.replace("\\", "/")
        path = re.sub(r'^([A-Za-z]):', lambda m: f"/mnt/{m.group(1).lower()}", path)
        return path

    def __init__(self, base_url: str = "http://127.0.0.1:8000", audio_tag: str = "<chinese>"):
        self.base_url = base_url.rstrip("/")
        self.audio_tag = audio_tag

    def health_check(self, timeout: int = 120, interval: int = 2) -> bool:
        """等待 ASR 服务就绪，成功返回 True，超时返回 False"""
        base = self.base_url
        print(f"[MiMoLocalASR] ⏳ 等待服务就绪（{base}）...", end="", flush=True)
        for _ in range(timeout // interval):
            try:
                resp = requests.get(f"{base}/health", timeout=5)
                if resp.status_code == 200 and resp.json().get("status") == "ok":
                    print(f" {Color.GREEN}✅{Color.END}")
                    return True
            except requests.RequestException:
                pass
            print(".", end="", flush=True)
            time.sleep(interval)
        print(f" {Color.RED}❌{Color.END}")
        return False

    def recognize(self, audio_file_path: str) -> str:
        import os as _os
        if not _os.path.isfile(audio_file_path):
            raise FileNotFoundError(f"找不到音频文件: {audio_file_path}")

        # 将 Windows 路径转为 WSL 路径（服务跑在 WSL 中）
        wsl_path = self._to_wsl_path(audio_file_path)
        print(f"[MiMoLocalASR] 正在调用本地 ASR 服务: {audio_file_path}")
        print(f"   → WSL 路径: {wsl_path}")

        try:
            resp = requests.post(
                f"{self.base_url}/asr",
                json={"audio_path": wsl_path, "audio_tag": self.audio_tag},
                timeout=600,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.ConnectionError:
            raise RuntimeError(f"[MiMoLocalASR] 无法连接到服务 {self.base_url}，请确认服务已启动")
        except requests.HTTPError as e:
            detail = ""
            try:
                detail = f" ({e.response.json()})"
            except Exception:
                pass
            raise RuntimeError(f"[MiMoLocalASR] 服务返回错误: {e}{detail}")

        text = data.get("text", "")
        duration_ms = data.get("duration_ms", 0)
        print(f"{Color.GREEN}[MiMoLocalASR] 识别完成 (音频 {duration_ms}ms):\n{text}")
        return text


class DoubaoASR(BaseASR):
    """
    使用火山引擎 (豆包) 大模型录音文件识别标准版 API 进行语音识别。
    """

    def __init__(self, model_name: str = "bigmodel"):
        self.api_key = os.environ.get("DOUBAO_API_KEY")
        if not self.api_key:
            raise ValueError("未检测到 DOUBAO_API_KEY 环境变量，请先设置火山引擎 API Key。")

        self.model_name = model_name
        # 默认使用豆包录音文件识别模型2.0的资源ID
        self.resource_id = "volc.seedasr.auc"

    def recognize(self, audio_file_path: str) -> str:
        print(f"[DoubaoASR] 正在处理音频文件: {audio_file_path}")

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

            print(f"[DoubaoASR] 正在提交异步转录任务...")
            
            submit_url = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/submit"
            task_id = str(uuid.uuid4())

            headers = {
                "X-Api-Key": self.api_key,
                "X-Api-Resource-Id": self.resource_id,
                "X-Api-Request-Id": task_id,
                "X-Api-Sequence": "-1",
                "Content-Type": "application/json"
            }

            request_payload = {
                "user": {
                    "uid": "doubao_asr_user"
                },
                "audio": {
                    "url": signed_url,
                    "format": "mp3"
                },
                "request": {
                    "model_name": self.model_name,
                    "enable_itn": True,
                    "enable_punc": True
                }
            }

            response = requests.post(submit_url, data=json.dumps(request_payload), headers=headers)
            
            if response.headers.get("X-Api-Status-Code") != "20000000":
                raise RuntimeError(f"提交任务失败! Headers: {response.headers}, Body: {response.text}")

            x_tt_logid = response.headers.get("X-Tt-Logid", "")
            print(f"[DoubaoASR] 任务已提交，task_id: {task_id}。正在等待任务完成...")

            query_url = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/query"
            query_headers = {
                "X-Api-Key": self.api_key,
                "X-Api-Resource-Id": self.resource_id,
                "X-Api-Request-Id": task_id,
                "X-Tt-Logid": x_tt_logid,
                "Content-Type": "application/json"
            }

            # 轮询查询结果
            while True:
                query_response = requests.post(query_url, json={}, headers=query_headers)
                code = query_response.headers.get("X-Api-Status-Code", "")
                
                if code == "20000000":  # 任务完成
                    print(f"{Color.GREEN}[DoubaoASR] 任务已完成！正在提取转录结果...")
                    res_data = query_response.json()
                    text = res_data.get("result", {}).get("text", "")
                    print(f"{Color.GREEN}[DoubaoASR 输出]:\n{text}")
                    return text
                elif code in ["20000001", "20000002"]:  # 处理中或排队中
                    time.sleep(3)
                else:
                    raise RuntimeError(f"任务执行失败! Code: {code}, Headers: {query_response.headers}, Body: {query_response.text}")

        finally:
            # 清理可能生成的临时压缩文件
            if is_temp and processed_audio_path and os.path.exists(processed_audio_path):
                os.remove(processed_audio_path)
