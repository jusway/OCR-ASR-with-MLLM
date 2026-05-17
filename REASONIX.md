# OCR-ASR-with-MLLM

## Stack

- **Language** — Python 3.11+
- **ASR** — Qwen3ASRFlashFiletrans (via dashscope)
- **LLM engines** — DeepSeek / NVIDIA NIM / Doubao / MiMo / Gemini (via openai or google-genai SDK)
- **Audio** — pydub (MP3 processing)
- **OSS** — oss2 (cloud upload)

## Layout

- `core/pipeline.py` — 5-step audio processing pipeline: ASR → proofread → loss-check → precision positioning → extract source text → metadata
- `core/text_engine.py` — LLM wrappers (GeminiText, DoubaoText, MiMoText, NvidiaText, DeepSeekText)
- `core/asr_engine.py` — ASR engine wrapper (Qwen3ASRFlashFiletrans)
- `core/utils.py` — color codes, audio compressor, chunker, PDF loader, OSS uploader
- `一键生成校对稿.py` — **main entry point**; edit here to configure folder, model, prompts
- `一键批量转录.py` — batch transcription without proofreading
- `常用脚本/` — file-management helpers (rename, organize, pack)
- `摩诃止观-定智法师 2017/` — lecture audio + per-session output (逐字稿, 校对稿, 精准原文, 思维链, 原文.txt)
- `暂时不用/` — archived scripts
- `trash/` — deprecated scripts

## Pipeline steps (in `一键生成校对稿.py`)

| Step | Description | Output file |
|------|-------------|-------------|
| 1 | ASR transcription (cached) | `{name}_逐字稿.md` |
| 2 | Proofread with reference (cached) | `{name}_校对稿.md` |
| 3 | Information loss check (cached) | `丢失信息检查_{有/无遗漏}.md` |
| 4 | Precision source-text positioning (cached) | `{name}_精准原文.md` |
| 5 | Model-based source extraction (cached) | `原文.txt` |

Each step skips if its output file already exists.

## Env vars required

- `DEEPSEEK_API_KEY` — for DeepSeekText
- `NVIDIA_API_KEY` — for NvidiaText
- `ARK_API_KEY` — for DoubaoText
- `MIMO_API_KEY` — for MiMoText

## DeepSeek thinking mode

Pass `enable_thinking: True, reasoning_effort: "max"` in the model kwargs.
Thinking mode disables `temperature` / `top_p` silently.
Reasoning content is saved per-step as `思维链_{步骤名}.txt` in the session folder.

## Conventions

- Script filenames are Chinese descriptions of their purpose
- Per-session output dir structure: one folder per lecture containing `.mp3` + generated `.md` / `.txt` files
- All prompts are configurable templates in `一键生成校对稿.py` (text_system_prompt, precision_user_prompt_template, etc.)
- Omission-check marker (case-sensitive): `【无遗漏信息】`

## Watch out for

- **LF→CRLF** — Windows Git warns "LF will be replaced by CRLF"; safe to ignore
- **`原文.txt` extracted by model** — no longer regex-based; extraction prompt in `extraction_user_prompt_template`
- **No `core/__init__.py`** — `core` is a namespace package; ensure `PYTHONPATH` includes project root
- **No `pyproject.toml`** — deps in bare `requirements.txt`; no pinned versions
