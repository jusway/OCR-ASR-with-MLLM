# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

This is a personal toolkit that converts long Buddhist-lecture audio recordings into proofread written drafts. It chains an ASR engine (verbatim transcription) with a text engine (refinement) and a verifier pass (information-loss check). There is no test suite, no build step, no CI — the project is run by executing one of the top-level Chinese-named scripts directly.

## Common Commands

```powershell
pip install -r requirements.txt   # also requires ffmpeg installed system-wide

python 一键生成书面稿.py              # full pipeline: ASR → 书面稿 → loss check → metadata
python 一键润色与检查.py              # text-only pipeline starting from an existing 逐字稿
python 信息损失检查.py                 # standalone loss check on an existing 逐字稿/书面稿 pair
```

All three scripts are launched without CLI args — edit the hard-coded constants in their `if __name__ == "__main__"` block before running.

## Architecture

### Engine pattern (core/)
All ASR and text providers inherit from abstract bases:
- `core/asr_engine.py` — `BaseASR` with `recognize(audio_path) -> str`. Implementations: `Qwen3ASRFlashFiletrans` (DashScope SDK), `DoubaoASR` (Volcengine HTTP API). Both compress audio to 16 kHz mono mp3 via `AudioCompressor`, upload to Alibaba OSS via `OSSAudioUploader`, then submit an async transcription job and poll.
- `core/text_engine.py` — `BaseTextModel` with `generate()` and `generate_stream()`. Implementations: `GeminiText` (google-genai), `DoubaoText` (OpenAI-compatible Ark endpoint), `MiMoText` (raw HTTP). Gemini calls have a built-in 5x retry on `APIError`.
- `core/ocr_engine.py` — `GeminiOCR` for PDF page → markdown (used standalone; not yet wired into the pipeline scripts).
- `core/utils.py` — `AudioCompressor`, `AudioChunker` (ffmpeg-based, used for chunked workflows), `OSSAudioUploader` (V4-signed URLs, dedups by object key), `PDFLoader`, `Color` (ANSI codes for console output).

When adding a new ASR or text provider, subclass the appropriate base class — the pipelines depend only on the abstract interface.

### Pipeline scripts (top level)
- `一键生成书面稿.py` — `AudioProcessingPipeline` runs 4 steps: transcript → 书面稿 → loss-check report → metadata header. Default engines: `Qwen3ASRFlashFiletrans` + `GeminiText`.
- `一键润色与检查.py` — `GeminiProcessingPipeline`, same shape but skips the ASR step (consumes an existing `*_逐字稿.md`).
- `信息损失检查.py` — standalone verifier on an existing 逐字稿/书面稿 pair; uses Gemini only.

### Output filename convention (load-bearing)
Per audio file `<base>.mp3`, the pipelines write into the same parent directory:
- `<base>_逐字稿.md` — verbatim transcript
- `<base>_书面稿.md` — refined draft
- `<base>_丢失信息检查.md` — verifier report

**Caching is filename-based**: each pipeline step checks `Path.exists()` and skips the (expensive) API call if the file is already there. To force a re-run, delete the corresponding `.md` file. Do not rename these suffixes — `信息损失检查.py` and the metadata-insertion step both pattern-match on them.

### Per-lecture data folder layout
Sample under `摩诃止观-久仁法师/摩诃止观NNN/`:
- `摩诃止观NNN.mp3` — source audio
- `NNN原文模糊范围.txt` — fuzzy reference text (the source scripture being lectured on); fed to the text engine as `【原文参考】` so it can correct proper nouns
- `摩诃止观NNN_精准原文.txt` — optional exact-text reference
- generated `_逐字稿.md` / `_书面稿.md` / `_丢失信息检查.md`

The fuzzy reference path is required by both pipeline scripts.

## Environment Variables

Set whichever are needed for the engines you instantiate:
- `DASHSCOPE_API_KEY` — Qwen ASR
- `OSS_ACCESS_KEY_ID`, `OSS_ACCESS_KEY_SECRET` — OSS uploader (bucket `asr-data-fofa`, region `cn-huhehaote`, hard-coded in `utils.py`)
- `ARK_API_KEY` — Doubao text (Volcengine Ark)
- `DOUBAO_API_KEY` — Doubao ASR
- `MIMO_API_KEY` — MiMo text
- `GOOGLE_API_KEY` — Gemini

`OSSAudioUploader` deliberately disables system proxies (`session.proxies = {}`, `trust_env = False`) — don't re-enable them, the original author hit proxy issues hitting OSS direct.

## Conventions

- Use `pathlib.Path`; always pass `encoding="utf-8"` to `open()` (Chinese content throughout).
- Refinement prompts enforce a strict quoting rule: any text matching the `【原文参考】` must be wrapped in `『』`, never `""`. If you edit the system prompts, preserve this constraint.
- `trash/` and `暂时不用/` hold deprecated/experimental scripts — don't import from them; treat them as archive.
- `GEMINI.md` is the equivalent guidance file for the Gemini CLI; keep it in sync if you change architecture.
