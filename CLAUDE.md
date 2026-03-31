# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OCR-ASR-with-MLLM 是一个佛教音频转录流水线工具，将长音频（讲法录音）转换为书面文稿。核心流程：音频定位原文范围 → 切片并发 ASR 转录 → 汇总逐字稿 → LLM 整理为书面稿。另有 PDF OCR 功能用于佛典图像转文字。

## Dependencies

```bash
pip install -r requirements.txt
```

Also requires `ffmpeg` in PATH (used by `pydub` for audio preprocessing).

## Environment Variables

- `MIMO_API_KEY` — Xiaomi MiMo API key (for MiMoASR / MiMoText)
- `DASHSCOPE_API_KEY` — Alibaba DashScope API key (for QwenASR)
- Gemini uses `google-genai` SDK, configured via `GOOGLE_API_KEY` or default credentials

## Running

Main entry point is `音频得到书面稿.py`. It's an interactive script that prompts for engine selection and metadata input:

```bash
python 音频得到书面稿.py
```

Individual core modules can also be run standalone for testing:

```bash
python -m core.asr_engine
python -m core.text_to_text_engine
python -m core.pdf_loader
```

## Architecture

### Core Modules (`core/`)

- **`asr_engine.py`** — ASR (语音识别) 引擎。定义 `BaseASR` 抽象基类，三个实现：
  - `GeminiASR` — 使用 `google-genai` SDK，音频上传到 Google Files API，流式生成，含重试和云端文件清理
  - `MiMoASR` — 使用 OpenAI 兼容接口，音频 Base64 编码内联发送，含 429 重试和内容审核拦截处理
  - `QwenASR` — 阿里云百炼接口，与 MiMoASR 结构相似，`modalities=["text"]` 仅输出文本
  - `AudioCompressor` — 音频超 50MB 时自动压缩（单声道 16kHz 32kbps MP3）

- **`audio_chunker.py`** — 音频切片器。用 ffmpeg 预处理（dynaudnorm + lowpass 8kHz），按指定时长切分并保留重叠区间，chunk 文件从 `chunk_001.mp3` 开始编号。如果输出目录已有切片文件则直接复用。

- **`text_to_text_engine.py`** — 文本生成引擎。定义 `BaseTextModel` 抽象基类，两个实现：`GeminiText`（google-genai SDK）和 `MiMoText`（OpenAI 兼容接口）。支持流式和非流式生成。

- **`pdf_loader.py`** — PDF 页面加载器，基于 `pdf2image`，按页码获取 PIL Image。

- **`gemini_ocr.py`** / **`gemini_debug_client.py`** — 旧版 Gemini OCR 客户端，使用第三方代理 API（aifuwu.icu）和原始 HTTP 请求，非 SDK 方式。

### Main Script (`音频得到书面稿.py`)

三个主要类：
- `BatchAudioTranscriber` — 切片 + 线程池并发 ASR + 缓存机制（已有 md 文件则跳过）
- `AudioTextLocator` — 将完整音频和模糊原文提交给 ASR，精确定位讲法对应的原文段落
- `AudioProcessingPipeline` — 端到端流水线，三步：定位 → 转录 → 书面化，最后交互式收集元数据（年份、作者、原文）并写入最终文档

### Data Directory (`摩诃止观-久仁法师/`)

按音频编号组织，每个目录包含：音频 mp3、原文模糊范围 txt、切片 chunks 目录、逐字稿 md、书面稿 md。

## Key Patterns

- 所有 ASR/Text 引擎使用策略模式：抽象基类 → 具体实现，运行时通过 `input()` 选择
- 重试机制采用指数退避（`2^n * 5` 秒），ASR 默认 5-6 次重试
- Gemini 路径有中文编码问题的 workaround：复制到纯英文临时文件再上传
- 音频切片缓存通过检查输出目录下是否已存在 `chunk_*.mp3` 文件实现；逐字稿缓存通过检查 `.md` 文件内容长度是否大于 header 长度实现
