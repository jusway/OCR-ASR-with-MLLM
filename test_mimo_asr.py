"""
MiMo-V2.5-ASR 本地引擎测试脚本
用法:
    python test_mimo_asr.py                                                  # 使用默认音频文件
    python test_mimo_asr.py --audio "D:\other\file.wav"                      # 指定其他音频
    python test_mimo_asr.py --tag "<english>"                                # 指定语言标签
    python test_mimo_asr.py --no-wait                                         # 不等待服务就绪
"""
import sys
import time
from pathlib import Path

from core.asr_engine import MiMoLocalASR

# ============================================================
# 配置
# ============================================================
DEFAULT_AUDIO = Path(
    r"D:\DATA\Project\OCR-ASR-with-MLLM\摩诃止观-定智法师 2017"
    r"\14正说分\2019年10月7日摩诃止观正说分14第七正修阴入界境道品调适005"
    r"\2019年10月7日摩诃止观正说分14第七正修阴入界境道品调适005.mp3"
)
BASE_URL = "http://172.29.1.0:8000"
AUDIO_TAG = "<chinese>"
WAIT_FOR_SERVICE = True

# ============================================================
# 执行
# ============================================================
if __name__ == "__main__":
    audio_path = DEFAULT_AUDIO
    audio_tag = AUDIO_TAG
    wait = WAIT_FOR_SERVICE

    # 简单解析命令行参数
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--audio" and i + 1 < len(sys.argv):
            audio_path = Path(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--tag" and i + 1 < len(sys.argv):
            audio_tag = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--no-wait":
            wait = False
            i += 1
        else:
            i += 1

    if not audio_path.exists():
        print(f"❌ 音频文件不存在: {audio_path}")
        sys.exit(1)

    print(f"🎤 音频文件: {audio_path}")
    print(f"🔗 服务地址: {BASE_URL}")
    print(f"🏷️  语言标签: {audio_tag}")
    print()

    asr = MiMoLocalASR(base_url=BASE_URL, audio_tag=audio_tag)

    if wait:
        print("⏳ 检查服务状态...")
        if not asr.health_check(timeout=30):
            print("❌ 服务未就绪，退出")
            sys.exit(1)
        print()

    print("📝 开始识别...")
    t0 = time.time()
    try:
        result = asr.recognize(str(audio_path))
        elapsed = time.time() - t0
        print()
        print(f"{'=' * 50}")
        print(f"✅ 识别结果")
        print(f"{'=' * 50}")
        print(result)
        print(f"{'=' * 50}")
        print(f"⏱️  耗时: {elapsed:.1f}秒")
    except Exception as e:
        print(f"❌ 识别失败: {e}")
        sys.exit(1)
