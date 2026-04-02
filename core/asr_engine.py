import os
import time
import json
import uuid
from abc import ABC, abstractmethod
import httpx

from core.utils import AudioCompressor, OSSAudioUploader, Color


class BaseASR(ABC):
    """
    ASR (语音识别) 抽象基类，规范统一的调用接口。
    """

    def _compress_audio(self, audio_file_path: str) -> tuple[str, bool]:
        """
        内部方法：将音频统一压缩为大模型标准 (16kHz 单声道 mp3)。
        返回: (处理后的音频路径, 是否生成了临时文件)
        """
        return AudioCompressor.compress(audio_file_path)

    @abstractmethod
    def recognize(self, system_prompt: str, prompt: str, audio_file_path: str) -> str:
        """
        根据提供的系统提示词、用户提示词和音频文件路径，返回转录文本。
        """
        pass


class Qwen3ASRFlashFiletrans(BaseASR):
    """
    使用 Qwen3-ASR-Flash-Filetrans 模型进行异步语音识别。
    """

    def __init__(self, model_name: str = "qwen3-asr-flash-filetrans"):
        self.api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("未检测到 DASHSCOPE_API_KEY 环境变量，请先设置阿里云 API Key。")

        self.model_name = model_name
        self.api_url_submit = "https://dashscope.aliyuncs.com/api/v1/services/audio/asr/transcription"
        self.api_url_query_base = "https://dashscope.aliyuncs.com/api/v1/tasks/"

    def recognize(self, system_prompt: str, prompt: str, audio_file_path: str) -> str:
        print(f"{Color.DARK_PURPLE}[Qwen3ASRFlash] 正在处理音频文件: {audio_file_path}")

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

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "X-DashScope-Async": "enable"
            }

            payload = {
                "model": self.model_name,
                "input": {
                    "file_url": signed_url
                },
                "parameters": {
                    "channel_id": [0],
                    "enable_itn": False
                }
            }

            print(f"{Color.DARK_PURPLE}[Qwen3ASRFlash] 正在提交异步转录任务...")
            
            with httpx.Client(timeout=30.0) as client:
                submit_resp = client.post(self.api_url_submit, headers=headers, json=payload)
                
                if submit_resp.status_code != 200:
                    raise RuntimeError(f"提交任务失败! HTTP code: {submit_resp.status_code}, Response: {submit_resp.text}")

                resp_data = submit_resp.json()
                output = resp_data.get("output")
                if not output or "task_id" not in output:
                    raise RuntimeError(f"提交任务失败，未获取到 task_id: {resp_data}")

                task_id = output["task_id"]
                print(f"{Color.DARK_PURPLE}[Qwen3ASRFlash] 任务已提交，task_id: {task_id}。正在轮询任务状态...")

                finished = False
                while not finished:
                    time.sleep(3)
                    query_url = self.api_url_query_base + task_id
                    query_resp = client.get(query_url, headers=headers)

                    if query_resp.status_code != 200:
                        print(f"{Color.RED}[Qwen3ASRFlash] 查询任务状态失败! HTTP code: {query_resp.status_code}")
                        continue

                    query_data = query_resp.json()
                    output = query_data.get("output")
                    if output and "task_status" in output:
                        status = output["task_status"]
                        
                        if status.upper() == "SUCCEEDED":
                            finished = True
                            print(f"{Color.GREEN}[Qwen3ASRFlash] 任务已完成！正在提取转录结果...")
                            
                            results = output.get("results", [])
                            if not results:
                                print(f"{Color.RED}[Qwen3ASRFlash] 警告：未找到 results 字段。原始 output: {output}")
                                return ""
                                
                            transcription_url = results[0].get("transcription_url")
                            if transcription_url:
                                # 下载结果 JSON
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
                                text = results[0].get("text", "")
                                print(f"{Color.GREEN}[Qwen3ASRFlash 输出]:\n{text}")
                                return text
                                
                        elif status.upper() == "FAILED":
                            finished = True
                            raise RuntimeError(f"任务执行失败: {query_data}")
                        elif status.upper() == "UNKNOWN":
                            finished = True
                            raise RuntimeError(f"任务状态未知: {query_data}")
                        else:
                            # RUNNING, PENDING 等状态，继续等待
                            pass

        finally:
            # 清理可能生成的临时压缩文件
            if is_temp and processed_audio_path and os.path.exists(processed_audio_path):
                os.remove(processed_audio_path)


def clean_repeated_text(text: str) -> str:
    """
    简单的后处理：移除连续重复的相同行（常见于ASR在静音处的幻觉）。
    如果某一行与上一行完全相同，则跳过不输出。
    """
    lines = text.split('\n')
    if not lines:
        return text
    
    cleaned = []
    for line in lines:
        stripped_line = line.strip()
        # 如果是空行，直接保留
        if not stripped_line:
            cleaned.append(line)
            continue
            
        # 如果当前行和上一行非空内容完全一样，说明是复读机幻觉，跳过
        if cleaned and stripped_line == cleaned[-1].strip():
            continue
            
        cleaned.append(line)
        
    return '\n'.join(cleaned)
