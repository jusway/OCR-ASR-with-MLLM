import os
from pathlib import Path
from pydub import AudioSegment

from core.asr_engine import Qwen3ASRFlashFiletrans
from core.text_engine import GeminiText, NvidiaText
from core.utils import Color


class AudioProcessingPipeline:
    """端到端音频处理流水线：整段转录 -> 书面化 -> 信息丢失检查 -> 补充元数据"""

    def __init__(self, asr_engine, text_engine):
        self.asr_engine = asr_engine
        self.text_engine = text_engine

    def _get_audio_duration(self, audio_path):
        audio = AudioSegment.from_mp3(audio_path)
        duration_seconds = int(audio.duration_seconds)
        hours, remainder = divmod(duration_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def run(self, 
            audio_path, 
            fuzzy_text_path, 
            text_sys_prompt,
            text_user_prompt_template,
            verifier_sys_prompt,
            verifier_user_prompt_template,
            year="",
            author=""):
        
        audio_path_obj = Path(audio_path)
        base_name = audio_path_obj.stem
        parent_dir = audio_path_obj.parent

        transcript_filename = parent_dir / f"{base_name}_逐字稿.md"
        final_output_filename = parent_dir / f"{base_name}_书面稿.md"
        verification_filename = parent_dir / f"{base_name}_丢失信息检查.md"

        # 读取模糊范围的原文
        with open(fuzzy_text_path, "r", encoding="utf-8") as f:
            fuzzy_reference_text = f.read()

        # ==========================================
        # 步骤 1: 整段音频直接转录 (带缓存)
        # ==========================================
        print(f"\n{Color.DARK_PURPLE}[1/4] 开始处理整段音频转录...")
        if transcript_filename.exists():
            print(f"{Color.GREEN}✅ 检测到已存在逐字稿，直接复用: {transcript_filename.name}")
            with open(transcript_filename, "r", encoding="utf-8") as f:
                transcript_text = f.read()
        else:
            print(f"{Color.DARK_PURPLE}正在调用 ASR 引擎生成逐字稿...")
            transcript_text = self.asr_engine.recognize(audio_path)

            with open(transcript_filename, "w", encoding="utf-8") as f:
                f.write(transcript_text)
            print(f"{Color.GREEN}逐字稿已保存至: {transcript_filename}")

        # ==========================================
        # 步骤 2: 整理为书面稿 (带缓存)
        # ==========================================
        print(f"\n{Color.DARK_PURPLE}[2/4] 开始处理书面稿...")
        if final_output_filename.exists():
            print(f"{Color.GREEN}✅ 检测到已存在书面稿，直接复用: {final_output_filename.name}")
            with open(final_output_filename, "r", encoding="utf-8") as f:
                written_text = f.read()
        else:
            print(f"{Color.DARK_PURPLE}正在使用 {self.text_engine.model_name} 将逐字稿整理为书面稿，请耐心等待...")
            user_prompt = text_user_prompt_template.format(
                fuzzy_reference_text=fuzzy_reference_text,
                transcript_text=transcript_text
            )

            written_text = self.text_engine.generate(text_sys_prompt, user_prompt)

            with open(final_output_filename, "w", encoding="utf-8") as f:
                f.write(written_text)
            print(f"{Color.GREEN}书面稿已保存至: {final_output_filename}")

        # ==========================================
        # 步骤 3: 信息丢失检查 (带缓存)
        # ==========================================
        print(f"\n{Color.DARK_PURPLE}[3/4] 开始检查信息丢失情况...")

        transcript_len = len(transcript_text)
        written_len = len(written_text)
        ratio = (written_len / transcript_len * 100) if transcript_len > 0 else 0.0
        print(f"{Color.ORANGE}📊 字数统计：书面稿 {written_len} / 逐字稿 {transcript_len} = {ratio:.1f}%")

        if verification_filename.exists():
            print(f"{Color.GREEN}✅ 检测到已存在检查报告，直接复用: {verification_filename.name}")
            with open(verification_filename, "r", encoding="utf-8") as f:
                verification_report = f.read()
        else:
            print(f"{Color.DARK_PURPLE}正在对比逐字稿与书面稿，检查是否有关键信息丢失...")
            verifier_user_prompt = verifier_user_prompt_template.format(
                transcript_text=transcript_text,
                written_text=written_text
            )

            verification_report = self.text_engine.generate(verifier_sys_prompt, verifier_user_prompt)

            with open(verification_filename, "w", encoding="utf-8") as f:
                f.write(verification_report)
            print(f"{Color.GREEN}检查报告已保存至: {verification_filename}")
        
        print(f"{Color.ORANGE}--- 信息丢失检查报告摘要 ---")
        # 只打印前10行，避免控制台过长
        print("\n".join(verification_report.split("\n")[:10]))
        if len(verification_report.split("\n")) > 10:
            print("...")
        print(f"{Color.ORANGE}----------------------------")

        # ==========================================
        # 步骤 4: 补充元数据并插入书面稿
        # ==========================================
        print(f"\n{Color.DARK_PURPLE}[4/4] 准备补充元数据信息...")
        with open(final_output_filename, "r", encoding="utf-8") as f:
            current_written_text = f.read()
            
        if current_written_text.strip().startswith("> 标题："):
            print(f"{Color.GREEN}✅ 检测到书面稿已包含元数据，跳过元数据。")
        else:
            duration = self._get_audio_duration(audio_path)
            
            metadata_header = (
                f"> 标题：{base_name}\n"
                f"> 时间：{year}\n"
                f"> 时长：{duration}\n"
                f"> 作者：{author}\n"
                "\n"
                "---\n\n"
            )
            
            final_content = metadata_header + current_written_text
            with open(final_output_filename, "w", encoding="utf-8") as f:
                f.write(final_content)
            print(f"{Color.GREEN}✅ 元数据已成功插入书面稿开头！")

        print(f"\n{Color.GREEN}===========================================")
        print("🎉 全部流程处理完成！")
        print(f"💡 最终文件列表：")
        print(f"  - 逐字稿: {transcript_filename.name}")
        print(f"  - 书面稿: {final_output_filename.name}")
        print(f"  - 检查报告: {verification_filename.name}")
        print(f"===========================================\n")


if __name__ == "__main__":
    # ============================================================
    # ================ 用户配置区（按需修改即可） ================
    # ============================================================

    # 1) 任务文件夹（脚本会自动从中找 .mp3 与原文 .txt）
    folder_path = Path("摩诃止观-久仁法师/摩诃止观003")

    # 2) 元数据（写入书面稿头部）
    year = "null"
    author = "久仁法师"

    # 3) ASR 引擎模型名
    asr_model_name = "qwen3-asr-flash-filetrans"

    # 4) 文本模型：想换模型只改 SELECTED_MODEL 这一行的 key
    #    新增模型：照样追加一行 "别名": (引擎类, {"model_name": "...", ...其它初始化参数}),
    AVAILABLE_MODELS = {
        "kimi":     (NvidiaText, {"model_name": "moonshotai/kimi-k2-instruct-0905", "temperature": 1.0, "top_p": 1.0}),
        "minimax":  (NvidiaText, {"model_name": "minimaxai/minimax-m2.7",           "temperature": 1.0, "top_p": 1.0}),
        "deepseek": (NvidiaText, {"model_name": "deepseek-ai/deepseek-v4-pro",      "temperature": 1.0, "top_p": 1.0}),
        "gemini":   (GeminiText, {"model_name": "gemini-3.1-pro-preview",           "temperature": 1.0}),
    }
    SELECTED_MODEL = "deepseek"

    # 5) 书面化提示词
    text_system_prompt = (
        # 角色设定
        "您是一位极其严谨的佛学典籍校勘专家。您的唯一目标是将【口语逐字稿】转化为【书面文稿】，但必须做到「字斟句酌，一字不落」。\n\n"

        # 核心任务与边界（最重要的一段）
        "【修改边界：绝对的忠实】\n"
        "1. 允许的修改：只允许删除无意义的口癖（如“啊”、“那个”、“然后呢”）、修正结巴重复、调整严重语病以保证句子通顺。\n"
        "2. 禁止的修改（红线）：**绝对禁止**进行任何形式的“总结”、“概括”或“大意提取”！师父所讲的每一个排比句、每一个比喻、每一个实修细节（例如具体的操作方法）、每一个法相名词（如“三世佛母”、“智镜”等），必须 100% 完整保留。\n"
        "3. 宁可显得啰嗦，也绝不可丢失法义。\n\n"

        # 信息规范
        "【参考与校对】\n"
        "1. 【逐字稿】由于语音识别可能有错别字，请根据上下文和佛学知识进行修正，不要被机器翻译带偏。\n"
        "2. 必须高度重视【原文参考】。只要师父的讲解中引用了【原文参考】里的词句，**必须且只能**使用『』符号将其引起来（例如：如经文所说『如是我闻』）。\n\n"

        # 输出规范
        "【输出规范】\n"
        "1. 禁止使用 Markdown 标题（如 #, ##）或列表符号（如 -, *），直接输出纯文本，根据逻辑大意自然分段即可。\n"
        "2. 严格只输出书面稿正文，绝对不要说任何解释性或前言后语的废话。"
    )
    text_user_prompt_template = (
        "【原文参考】\n{fuzzy_reference_text}\n\n"
        "【逐字稿】\n{transcript_text}\n"
    )

    # 6) 信息丢失检查提示词
    verifier_system_prompt = (
        "你是一位严谨校对员。你的任务是对比【原始逐字稿】和【润色后的书面稿】，检查书面稿是否丢失了逐字稿中的关键信息。"
        "如果没有遗漏，直接输出【无遗漏信息】。"
        "如果有遗漏，请解释。"
    )
    verifier_user_prompt_template = (
        "【原始逐字稿】\n{transcript_text}\n\n"
        "【润色后的书面稿】\n{written_text}\n"
    )

    # ============================================================
    # ================ 以下为流程代码，无需改动 ==================
    # ============================================================

    if not folder_path.exists():
        print(f"{Color.RED}错误：找不到文件夹 {folder_path}")
        exit(1)

    # 自动寻找音频文件 (.mp3)，多个时优先和文件夹同名的
    audio_files = list(folder_path.glob("*.mp3"))
    if not audio_files:
        print(f"{Color.RED}错误：在 {folder_path} 下没找到 .mp3 文件")
        exit(1)
    audio_path_obj = next((f for f in audio_files if f.stem == folder_path.name), audio_files[0])
    audio_path = str(audio_path_obj)

    # 自动寻找原文参考文件 (.txt)，优先包含“原文”关键字
    txt_files = list(folder_path.glob("*原文*.txt"))
    if not txt_files:
        txt_files = list(folder_path.glob("*.txt"))
    if not txt_files:
        print(f"{Color.RED}错误：在 {folder_path} 下没找到参考原文 (.txt) 文件")
        exit(1)
    fuzzy_text_path = str(txt_files[0])

    print(f"{Color.GREEN}已自动识别任务：")
    print(f"  - 文件夹: {folder_path}")
    print(f"  - 音频文件: {audio_path_obj.name}")
    print(f"  - 参考原文: {Path(fuzzy_text_path).name}")

    # 初始化引擎
    engine_cls, engine_kwargs = AVAILABLE_MODELS[SELECTED_MODEL]

    print(f"{Color.DARK_PURPLE}正在初始化 Qwen3ASRFlashFiletrans 引擎...")
    asr_engine = Qwen3ASRFlashFiletrans(model_name=asr_model_name)

    print(f"{Color.DARK_PURPLE}正在初始化 {engine_cls.__name__} 引擎 ({SELECTED_MODEL} -> {engine_kwargs['model_name']})...")
    text_engine = engine_cls(**engine_kwargs)

    # 跑流水线
    pipeline = AudioProcessingPipeline(
        asr_engine=asr_engine,
        text_engine=text_engine
    )

    pipeline.run(
        audio_path=audio_path,
        fuzzy_text_path=fuzzy_text_path,
        text_sys_prompt=text_system_prompt,
        text_user_prompt_template=text_user_prompt_template,
        verifier_sys_prompt=verifier_system_prompt,
        verifier_user_prompt_template=verifier_user_prompt_template,
        year=year,
        author=author
    )
