"""
一键生成校对稿 — 稳扎版
四步流水线：初次校对 → 精准原文 → 最终校对 → 正文提取
每步单一职责，输入干净，输出稳定。
"""

from pathlib import Path

from config.models import AVAILABLE_MODELS
from core.stable_proofread_pipeline import TwoPassProofreadPipeline

# ============================================================
# ================ 以下为全部配置，按需修改 ===================
# ============================================================

# 1) 任务文件夹（脚本自动找 .mp3）
folder_path = Path(r"摩诃止观-定智法师 2017/05正说分第二释名/2018年5月14日摩诃止观05正说分第二释名003")

# 2) 元数据（写入校对稿头部）
year = "2018"
author = "定智法师"

# 3) ASR 引擎模型名
asr_model_name = "qwen3-asr-flash-filetrans"

# 4) 文本模型配置（定义在 config/models.py）
#    Pro = deepseek-v4-pro, Flash = deepseek-v4-flash
#    思考模式 effort: max / high；非思考模式无需 reasoning_effort
#    加新模型只改 config/models.py，不改入口脚本

# ── 步骤模型配置 ──
DRAFT_SELECTED_MODEL = "flash-nothink"      # ① 初次校对
REFINE_SELECTED_MODEL = "flash-nothink"        # ② 精准原文
FINAL_SELECTED_MODEL = "pro-nothink"      # ③ 最终校对
EXTRACT_SELECTED_MODEL = "flash-nothink"       # ④ 正文提取


# 优秀案例（从文档读取，作为 one-shot 示例）
example_path = Path("docs/优秀案例.md")
example_text = example_path.read_text("utf-8").strip() if example_path.exists() else ""

# 精准原文案例（用于步骤②）
refine_example_path = Path("docs/精准原文案例.md")
refine_example_text = refine_example_path.read_text("utf-8").strip() if refine_example_path.exists() else ""

# ============================================================
# ── 步骤 ①：初次校对稿（不参考原文，只把 ASR 转成书面语）──
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
直接输出校对稿的正文内容。
"""

# ============================================================
# ── 步骤 ②：精准原文（从模糊原文中砍掉没讲到的，只留讲到的）──
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
参考案例，从模糊原文参考中截取本讲稿所讲解的《摩诃止观辅行传弘决辑注》的范围。
只保留讲稿讲到了的原文内容，没讲到的部分砍掉。
直接输出截取后的原文（保留**包裹格式）。
"""

# ============================================================
# ── 步骤 ③：最终校对稿（用精准原文校对，内嵌自查遗漏）────
# ============================================================

final_system_prompt = """\
你是一位编辑。你擅长拿着精准原文校对录音稿件，整理成书面语校对稿，不丢信息。这一点一直是你的骄傲。"""
final_user_prompt_template = """\
【优秀案例】
{example_text}
【精准原文】
{precision_text}
【录音稿件】
{transcript_text}
## 背景
录音稿件是一位师父在讲解《摩诃止观辅行传弘决辑注》的记录。
录音稿件是ASR转录的，包含错别字、同音字和语气词。
精准原文是师父本次所讲的《摩诃止观辅行传弘决辑注》的原文范围，供参考。
## 任务
参考优秀案例的风格，根据精准原文校对录音稿件。
在校对完成后，请回看录音稿件，逐句核对是否有遗漏信息。
如果发现有遗漏，直接补充进校对稿中，不要单独说明。
确认无遗漏后，在文末加上：【已核查无遗漏】
输出完整的校对稿正文。
"""

# ============================================================
# ── 步骤 ④：提取摩诃止观正文 ─────────────────────────────
# ============================================================

extract_system_prompt = """你是一位严谨的文献整理员。"""
extract_user_prompt_template = """\
【校对稿】
{written_text}
【精准原文】
{precision_text}
## 背景
讲稿是一位师父在讲解《摩诃止观辅行传弘决辑注》。
校对稿是师父所讲内容的书面语整理。
精准原文是师父本次讲解的《摩诃止观辅行传弘决辑注》的原文范围。
## 任务
找到【校对稿】的内容对应在讲解【精准原文】中**摩诃止观正文**的哪些句子（按顺序）。
(讲稿最后如果存在只是提了一下的正文，但并没有开讲，这是下节课要讲的，本节课不纳入)
然后把这些**摩诃止观正文**按顺序直接拼接输出。
"""

# ============================================================
# ── 执行 ────────────────────────────────────────────────
# ============================================================

if __name__ == "__main__":
    # 文件发现
    if not folder_path.exists():
        print(f"错误：找不到文件夹 {folder_path}")
        exit(1)

    audio_files = list(folder_path.glob("*.mp3"))
    if not audio_files:
        print(f"错误：在 {folder_path} 下没找到 .mp3 文件")
        exit(1)
    audio_path_obj = next((f for f in audio_files if f.stem == folder_path.name), audio_files[0])

    print(f"已自动识别任务：")
    print(f"  - 文件夹: {folder_path}")
    print(f"  - 音频文件: {audio_path_obj.name}")

    # 初始化引擎
    def _init(label, key):
        cls, kwargs = AVAILABLE_MODELS[key]
        print(f"正在初始化 {cls.__name__} 引擎 ({key} -> {kwargs['model_name']})...")
        return cls(**kwargs)

    from core.asr_engine import Qwen3ASRFlashFiletrans

    print("正在初始化 Qwen3ASRFlashFiletrans 引擎...")
    asr_engine = Qwen3ASRFlashFiletrans(model_name=asr_model_name)

    draft_engine = _init("① 初次校对", DRAFT_SELECTED_MODEL)
    refine_engine = _init("② 精准原文", REFINE_SELECTED_MODEL)
    final_engine = _init("③ 最终校对", FINAL_SELECTED_MODEL)
    extract_engine = _init("④ 正文提取", EXTRACT_SELECTED_MODEL)

    # 跑流水线
    pipeline = TwoPassProofreadPipeline(
        asr_engine=asr_engine,
        draft_engine=draft_engine,
        refine_engine=refine_engine,
        final_engine=final_engine,
        extract_engine=extract_engine,
    )

    pipeline.run(
        audio_path=str(audio_path_obj),
        draft_sys_prompt=draft_system_prompt,
        draft_user_prompt_template=draft_user_prompt_template,
        refine_sys_prompt=refine_system_prompt,
        refine_user_prompt_template=refine_user_prompt_template,
        final_sys_prompt=final_system_prompt,
        final_user_prompt_template=final_user_prompt_template,
        extract_sys_prompt=extract_system_prompt,
        extract_user_prompt_template=extract_user_prompt_template,
        refine_example_text=refine_example_text,
        example_text=example_text,
        year=year,
        author=author,
    )
