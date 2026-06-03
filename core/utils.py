import os
import uuid
import glob
import subprocess
import tempfile
from pathlib import Path
from pydub import AudioSegment
from pdf2image import convert_from_path, pdfinfo_from_path
from PIL import Image
import oss2


class Color:
    """控制台输出颜色代码"""
    RED = '\033[91m'
    GREEN = '\033[1;32m'      # 加粗的标准绿，最扎实清晰
    ORANGE = '\033[1;38;5;208m'  # 橙色（加粗），用于警告
    END = '\033[0m'  # 重置所有样式


class AudioCompressor:
    """音频压缩工具类，将音频统一压缩为 16kHz 单声道 mp3。"""

    @staticmethod
    def compress(audio_path: str) -> tuple[str, bool]:
        """
        将音频压缩为 16kHz 单声道 32k bitrate 的 mp3 格式。
        返回: (处理后的音频路径, 是否生成了临时文件)
        """
        file_size = os.path.getsize(audio_path)
        print(f"{Color.DARK_PURPLE}[AudioCompressor] 正在压缩音频 (原大小: {file_size / 1024 / 1024:.2f} MB)...")
        from pydub import AudioSegment

        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_channels(1).set_frame_rate(16000)

        fd, temp_compressed_path = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)

        audio.export(temp_compressed_path, format="mp3", bitrate="32k")
        print(f"{Color.GREEN}[AudioCompressor] 压缩完成，新文件大小: {os.path.getsize(temp_compressed_path) / 1024 / 1024:.2f} MB")

        return temp_compressed_path, True


class AudioChunker:
    def __init__(self, chunk_minutes=3, overlap_seconds=10):
        self.chunk_ms = chunk_minutes * 60 * 1000
        self.overlap_ms = overlap_seconds * 1000
        self.step = self.chunk_ms - self.overlap_ms

    def _preprocess_audio(self, input_path: str) -> str:
        ext = Path(input_path).suffix
        output_path = f"temp_processed_full_{uuid.uuid4().hex}{ext}"

        command = [
            'ffmpeg',
            '-y',
            '-i', input_path,
            '-af', 'dynaudnorm,lowpass=f=8000',
            output_path
        ]

        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return output_path

    def process(self, file_path, output_dir="output_chunks"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            existing_files = sorted(glob.glob(os.path.join(output_dir, "chunk_*.mp3")))
            if existing_files:
                return existing_files

        processed_full_audio_path = self._preprocess_audio(file_path)

        audio = AudioSegment.from_file(processed_full_audio_path)
        total_duration = len(audio)

        generated_files = []
        start = 0
        index = 1  # 从 1 开始数数

        while start < total_duration:
            end = min(start + self.chunk_ms, total_duration)
            chunk = audio[start:end]

            out_path = os.path.join(output_dir, f"chunk_{index:03d}.mp3")
            chunk.export(out_path, format="mp3")
            generated_files.append(out_path)

            start += self.step
            index += 1

        os.remove(processed_full_audio_path)

        return generated_files

    @staticmethod
    def extract_head_tail(audio_path: str, output_dir: str, minutes: int = 15) -> tuple[str, str]:
        """提取音频的前 minutes 分钟和后 minutes 分钟，返回 (head_path, tail_path)。"""
        os.makedirs(output_dir, exist_ok=True)

        audio = AudioSegment.from_file(audio_path)
        total_ms = len(audio)
        excerpt_ms = minutes * 60 * 1000

        head_path = os.path.join(output_dir, "head_excerpt.mp3")
        tail_path = os.path.join(output_dir, "tail_excerpt.mp3")

        # 前 minutes 分钟
        head = audio[:min(excerpt_ms, total_ms)]
        head.export(head_path, format="mp3")

        # 后 minutes 分钟
        tail_start = max(0, total_ms - excerpt_ms)
        tail = audio[tail_start:]
        tail.export(tail_path, format="mp3")

        return head_path, tail_path


class PDFLoader:
    # 加载pdf的某一页
    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"找不到文件: {self.pdf_path}")

        info = pdfinfo_from_path(str(self.pdf_path))
        self.total_pages = int(info["Pages"])

    def get_page(self, page_num: int, dpi: int = 300) -> Image.Image:
        if page_num < 1 or page_num > self.total_pages:
            raise ValueError(f"页码超出范围: {page_num} (共 {self.total_pages} 页)")

        images = convert_from_path(
            str(self.pdf_path),
            dpi=dpi,
            first_page=page_num,
            last_page=page_num
        )
        return images[0]

class OSSAudioUploader:
    """将音频文件上传到阿里云 OSS，返回签名 URL。"""

    def __init__(self, url_expiry_seconds: int = 3600):
        access_key_id = os.environ.get("OSS_ACCESS_KEY_ID")
        access_key_secret = os.environ.get("OSS_ACCESS_KEY_SECRET")
        bucket_name = "asr-data-fofa"
        endpoint = "https://oss-cn-huhehaote.aliyuncs.com"
        region = "cn-huhehaote"  # V4 签名需要指定 region

        missing = [k for k, v in {
            "OSS_ACCESS_KEY_ID": access_key_id,
            "OSS_ACCESS_KEY_SECRET": access_key_secret,
        }.items() if not v]
        if missing:
            raise ValueError(f"未检测到环境变量: {', '.join(missing)}，请先配置阿里云 OSS 凭证。")

        # 使用 AuthV4 替代默认的 Auth (V1 签名)
        auth = oss2.AuthV4(access_key_id, access_key_secret)
        # 初始化 Bucket 时传入 region 参数
        self.bucket = oss2.Bucket(auth, endpoint, bucket_name, region=region)
        self.bucket.session.proxies = {}  # 跳过系统代理，直连 OSS
        self.bucket.session.trust_env = False  # 忽略环境变量中的代理设置 (HTTP_PROXY/HTTPS_PROXY)
        self.prefix = "ocr-asr-tmp"
        self.url_expiry_seconds = url_expiry_seconds

    def upload(self, local_path: str, filename: str = None) -> tuple[str, str]:
        """上传本地文件到 OSS，如果同名文件已存在则直接复用，返回 (签名 URL, object_key)。"""
        if filename is None:
            filename = os.path.basename(local_path)
        object_key = f"{self.prefix}/{filename}"

        # 检查文件是否已经存在于 OSS 中
        if self.bucket.object_exists(object_key):
            print(f"{Color.GREEN}[OSS] 文件已存在，直接复用: {object_key}")
        else:
            self.bucket.put_object_from_file(object_key, local_path)
            print(f"{Color.GREEN}[OSS] 已上传: {object_key}")

        signed_url = self.bucket.sign_url("GET", object_key, self.url_expiry_seconds)
        return signed_url, object_key

    def delete(self, object_key: str):
        """根据 object_key 删除 OSS 上的文件。"""
        self.bucket.delete_object(object_key)
        print(f"{Color.DARK_PURPLE}[OSS] 已删除: {object_key}")

    def cleanup_all(self):
        """删除 prefix 下所有临时文件。"""
        keys = [obj.key for obj in oss2.ObjectIterator(self.bucket, prefix=f"{self.prefix}/")]
        if keys:
            self.bucket.batch_delete_objects(keys)
            print(f"{Color.DARK_PURPLE}[OSS] 已清理 {len(keys)} 个临时文件")


def ffprobe_duration(audio_path: str) -> str:
    """用 ffprobe 获取音频时长，返回 HH:MM:SS 格式字符串。"""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(audio_path)],
        capture_output=True, text=True
    )
    duration_seconds = int(float(result.stdout.strip()))
    hours, remainder = divmod(duration_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_duration(seconds: float) -> str:
    """将秒数格式化为 X分X秒 或 X秒。"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes > 0:
        return f"{minutes}分{secs}秒"
    return f"{secs}秒"


if __name__ == "__main__":
    audio_path = r"D:\DATA\Project\OCR-ASR-with-MLLM\摩诃止观-久仁法师\摩诃止观001\摩诃止观001.mp3"

    uploader = OSSAudioUploader()
    try:
        signed_url, object_key = uploader.upload(audio_path)
        print(f"签名 URL: {signed_url}")
        print(f"Object Key: {object_key}")
    except Exception as e:
        print(f"发生错误: {e}")
