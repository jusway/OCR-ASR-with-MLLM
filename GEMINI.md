# Project: OCR-ASR-with-MLLM

This project is a specialized toolkit designed to convert audio recordings of lectures (primarily Buddhist teachings) into high-quality, proofread written drafts. It leverages state-of-the-art ASR (Automatic Speech Recognition) and MLLMs (Multimodal Large Language Models) for transcription, text refinement, and terminology verification.

## 🏗️ Architecture & Core Components

The project is structured around a modular engine design in the `core/` directory:

### 1. ASR Engines (`core/asr_engine.py`)
Responsible for converting speech to text.
- **Qwen3ASRFlashFiletrans**: Utilizes Alibaba's Qwen3-ASR-Flash-Filetrans model via DashScope.
- **DoubaoASR**: Utilizes Bytedance's Doubao ASR API.
- **Common Logic**: Audio files are automatically compressed to 16kHz mono MP3s and uploaded to Alibaba OSS for processing.

### 2. Text Engines (`core/text_engine.py`)
Responsible for proofreading, formatting, and fact-checking.
- **GeminiText**: Integrates Google's Gemini models.
- **DoubaoText**: Integrates Bytedance's Doubao models.
- **MiMoText**: Integrates MiMo models via an OpenAI-compatible interface.

### 3. Utilities (`core/utils.py`)
- **AudioCompressor**: Standardizes audio for model compatibility.
- **OSSAudioUploader**: Handles secure uploads to Alibaba Cloud OSS.
- **PDFLoader**: Digitizes PDFs into images for potential OCR processing.
- **Color**: Provides formatted console output for progress tracking.

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- `ffmpeg` installed on the system (for audio processing).

### Installation
```powershell
pip install -r requirements.txt
```

### Environment Configuration
The following environment variables are required depending on the engines used:
- `DASHSCOPE_API_KEY`: Alibaba Cloud DashScope.
- `OSS_ACCESS_KEY_ID` & `OSS_ACCESS_KEY_SECRET`: Alibaba Cloud OSS credentials.
- `ARK_API_KEY`: Bytedance Ark API (Text).
- `DOUBAO_API_KEY`: Bytedance Doubao API (ASR).
- `MIMO_API_KEY`: MiMo API.
- `GOOGLE_API_KEY`: Google GenAI API.

### Usage
The primary entry points are:

1. **一键生成校对稿.py**: Full pipeline (Transcription → Refinement → Loss Check → Metadata).
2. **信息损失检查.py**: Standalone script to compare an existing `*_逐字稿.md` and `*_校对稿.md` for information loss.

#### Full Pipeline Step-by-Step (Internal to 一键生成校对稿.py):
1. **Transcription**: Generate a raw verbatim transcript.
2. **Refinement**: Convert the transcript into a readable written draft using a text engine.
3. **Information Loss Check**: Compare the verbatim transcript with the refined draft to ensure no key information or teachings were missed.
4. **Metadata Insertion**: Add headers (Title, Author, Year, etc.) to the final file.

```powershell
python 一键生成校对稿.py
python 信息损失检查.py
```

## 📜 Development Conventions

- **Engine Pattern**: All new ASR or Text engines should inherit from `BaseASR` or `BaseTextModel` in their respective files.
- **UTF-8 Encoding**: Always use `encoding="utf-8"` when reading or writing files.
- **Path Management**: Use `pathlib.Path` for cross-platform file path handling.
- **Caching**: The pipeline checks for existing intermediate files (e.g., `*_逐字稿.md`) to avoid redundant API calls.

## 📂 Directory Structure
- `core/`: Core engine implementations and utilities.
- `暂时不用/`: Archived or experimental scripts.
- `trash/`: Deprecated scripts and debug clients.
- `requirements.txt`: Project dependencies.
- `GEMINI.md`: Project documentation and CLI instructions (this file).
