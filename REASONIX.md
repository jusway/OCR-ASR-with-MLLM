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
- ============================================================
# ── 步骤 ②：精准原文 ───
# ============================================================

refine_system_prompt = """\
你是一位严谨的文献整理员。你擅长从杂乱的原文参考中，精准定位讲稿所对应的那部分原文。"""
refine_user_prompt_template = """\
【参考案例】
{refine_example_text}

【讲稿】
{draft_text}
【模糊原文参考】
{fuzzy_reference_text}
## 背景
讲稿是师父在讲解某一段《摩诃止观辅行传弘决辑注》。
模糊原文参考是我估摸着本次讲解《摩诃止观辅行传弘决辑注》的大致范围，从docx复制到md文件中所得。
模糊原文参考一般包含四部分：摩诃止观正文（一般是被**包裹）、摩诃止观科判（一般在摩诃止观正文前）、摩诃止观解释（一般在摩诃止观正文后）、注脚（一般是罗列在后）。
## 任务
从模糊原文参考中精准截取本讲稿所讲解的范围输出。
输出案例和格式参考"参考案例"。
直接输出结果，不说任何多余的话。
"""


好的，帮我创建这个档案馆，然后把这个提示词存一个文件，文件名字起的要见名知意
- ============================================================
# ── 步骤 ①：初次校对稿 ───
# ============================================================

draft_system_prompt = """\
你是一位编辑。你擅长将录音稿件整理成书面语的校对稿，而且不会丢失任何的信息。\
这一点一直是你的骄傲。"""
draft_user_prompt_template = """\
【优秀案例】
{example_text}
【录音稿件】
{transcript_text}
## 背景
录音稿件是一位师父在讲解《摩诃止观辅行传弘决辑注》的记录。
录音稿件是ASR转录的，包含错别字、同音字和语气词等。
## 任务
参考优秀案例的风格，将录音稿件整理成书面语的校对稿。
直接输出结果，不说任何多余的话。
"""

这个也帮我存一个
- ============================================================
# ── 步骤 ①：初次校对稿 ───
# ============================================================

draft_system_prompt = """\
你是一位编辑。你擅长将录音稿件整理成书面语的校对稿，而且不会丢失任何的信息。\
这一点一直是你的骄傲。"""
draft_user_prompt_template = """\
【优秀案例】
{example_text}
【录音稿件】
{transcript_text}
## 背景
录音稿件是一位师父在讲解《摩诃止观辅行传弘决辑注》的记录。
录音稿件是ASR转录的，包含错别字、同音字和语气词等。
## 任务
参考优秀案例的风格，将录音稿件整理成书面语的校对稿。
直接输出结果，不说任何多余的话。
"""

这个也帮我存一个
- ============================================================
# ── 步骤 ①：初次校对稿 ───
# ============================================================

draft_system_prompt = """\
你是一位编辑。你擅长将录音稿件整理成书面语的校对稿，而且不会丢失任何的信息。\
这一点一直是你的骄傲。"""
draft_user_prompt_template = """\
【优秀案例】
{example_text}
【录音稿件】
{transcript_text}
## 背景
录音稿件是一位师父在讲解《摩诃止观辅行传弘决辑注》的记录。
录音稿件是ASR转录的，包含错别字、同音字和语气词等。
## 任务
参考优秀案例的风格，将录音稿件整理成书面语的校对稿。
直接输出结果，不说任何多余的话。
"""

这个也帮我存一下
- ============================================================
# ── 步骤 ①：初次校对稿 ───
# ============================================================

draft_system_prompt = """\
你是一位编辑。你擅长将录音稿件整理成书面语的校对稿，而且不会丢失任何的信息。\
这一点一直是你的骄傲。"""
draft_user_prompt_template = """\
【优秀案例】
{example_text}
【录音稿件】
{transcript_text}
## 背景
录音稿件是一位师父在讲解《摩诃止观辅行传弘决辑注》的记录。
录音稿件是ASR转录的，包含错别字、同音字和语气词等。
## 任务
参考优秀案例的风格，将录音稿件整理成书面语的校对稿。
直接输出结果，不说任何多余的话。
"""

这个也帮我存一个文件
- ============================================================
# ── 步骤 ①：初次校对稿 ───
# ============================================================

draft_system_prompt = """\
你是一位编辑。你擅长将录音稿件整理成书面语的校对稿，而且不会丢失任何的信息。\
这一点一直是你的骄傲。"""
draft_user_prompt_template = """\
【优秀案例】
{example_text}
【录音稿件】
{transcript_text}
## 背景
录音稿件是一位师父在讲解《摩诃止观辅行传弘决辑注》的记录。
录音稿件是ASR转录的，包含错别字、同音字和语气词等。
## 任务
参考优秀案例的风格，将录音稿件整理成书面语的校对稿。
直接输出结果，不说任何多余的话。
"""

这个也帮我存个文件
