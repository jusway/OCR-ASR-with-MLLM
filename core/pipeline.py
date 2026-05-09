from pathlib import Path
from pydub import AudioSegment

from core.asr_engine import Qwen3ASRFlashFiletrans
from core.utils import Color


class AudioProcessingPipeline:
    """端到端音频处理流水线：转录 → 校对 → 精准定位 → 信息丢失检查 → 元数据"""

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
            fuzzy_reference_text="",
            text_sys_prompt="",
            text_user_prompt_template="",
            verifier_sys_prompt="",
            verifier_user_prompt_template="",
            precision_sys_prompt="",
            precision_user_prompt_template="",
            year="",
            author="",
            reference_text=""):

        audio_path_obj = Path(audio_path)
        base_name = audio_path_obj.stem
        parent_dir = audio_path_obj.parent

        transcript_filename = parent_dir / f"{base_name}_逐字稿.md"
        final_output_filename = parent_dir / f"{base_name}_校对稿.md"
        fuzzy_filename = parent_dir / "模糊原文.md"

        # 保存模糊原文为文件（始终以代码中的为准）
        if fuzzy_reference_text:
            with open(fuzzy_filename, "w", encoding="utf-8") as f:
                f.write(fuzzy_reference_text)

        # 步骤 1: 整段音频直接转录 (带缓存)
        print(f"\n{Color.DARK_PURPLE}[1/5] 开始处理整段音频转录...")
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

        # 步骤 2: 整理为校对稿 (带缓存)
        print(f"\n{Color.DARK_PURPLE}[2/5] 开始处理校对稿...")
        if final_output_filename.exists():
            print(f"{Color.GREEN}✅ 检测到已存在校对稿，直接复用: {final_output_filename.name}")
            with open(final_output_filename, "r", encoding="utf-8") as f:
                written_text = f.read()
        else:
            print(f"{Color.DARK_PURPLE}正在使用 {self.text_engine.model_name} 将逐字稿整理为校对稿，请耐心等待...")
            user_prompt = text_user_prompt_template.format(
                fuzzy_reference_text=fuzzy_reference_text,
                transcript_text=transcript_text
            )

            written_text = self.text_engine.generate(text_sys_prompt, user_prompt)

            with open(final_output_filename, "w", encoding="utf-8") as f:
                f.write(written_text)
            print(f"{Color.GREEN}校对稿已保存至: {final_output_filename}")

        # 步骤 3: 精准定位原文 (带缓存)
        precision_filename = parent_dir / f"{base_name}_精准原文.md"
        if precision_sys_prompt and precision_user_prompt_template:
            print(f"\n{Color.DARK_PURPLE}[3/5] 开始精准定位原文...")
            if precision_filename.exists():
                print(f"{Color.GREEN}✅ 检测到已存在精准原文，直接复用: {precision_filename.name}")
                with open(precision_filename, "r", encoding="utf-8") as f:
                    precision_text = f.read()
            else:
                print(f"{Color.DARK_PURPLE}正在根据校对稿定位精准原文范围...")
                precision_prompt = precision_user_prompt_template.format(
                    written_text=written_text,
                    fuzzy_reference_text=fuzzy_reference_text
                )
                precision_text = self.text_engine.generate(precision_sys_prompt, precision_prompt)
                with open(precision_filename, "w", encoding="utf-8") as f:
                    f.write(precision_text)
                print(f"{Color.GREEN}精准原文已保存至: {precision_filename}")
        else:
            precision_text = ""

        # 步骤 4: 信息丢失检查 (带缓存)
        print(f"\n{Color.DARK_PURPLE}[4/5] 开始检查信息丢失情况...")

        transcript_len = len(transcript_text)
        written_len = len(written_text)
        ratio = (written_len / transcript_len * 100) if transcript_len > 0 else 0.0
        print(f"{Color.ORANGE}📊 字数统计：校对稿 {written_len} / 逐字稿 {transcript_len} = {ratio:.1f}%")

        cached = list(parent_dir.glob(f"{base_name}_丢失信息检查_*.md"))
        if cached:
            verification_filename = cached[0]
            print(f"{Color.GREEN}✅ 检测到已存在检查报告，直接复用: {verification_filename.name}")
            verification_report = verification_filename.read_text("utf-8")
        else:
            print(f"{Color.DARK_PURPLE}正在对比逐字稿与校对稿，检查是否有关键信息丢失...")
            verifier_user_prompt = verifier_user_prompt_template.format(
                transcript_text=transcript_text,
                written_text=written_text,
                precision_text=precision_text
            )

            verification_report = self.text_engine.generate(verifier_sys_prompt, verifier_user_prompt)
            has_loss = "无遗漏信息" not in verification_report
            tag = "有遗漏" if has_loss else "无遗漏"
            verification_filename = parent_dir / f"{base_name}_丢失信息检查_{tag}.md"
            verification_filename.write_text(verification_report, "utf-8")
            print(f"{Color.GREEN}检查报告已保存至: {verification_filename}（{tag}）")

        print(f"{Color.ORANGE}--- 信息丢失检查报告摘要 ---")
        print("\n".join(verification_report.split("\n")[:10]))
        if len(verification_report.split("\n")) > 10:
            print("...")
        print(f"{Color.ORANGE}----------------------------")

        # 步骤 5: 补充元数据并插入校对稿
        print(f"\n{Color.DARK_PURPLE}[5/5] 准备补充元数据信息...")
        with open(final_output_filename, "r", encoding="utf-8") as f:
            current_written_text = f.read()

        if current_written_text.strip().startswith("> 标题："):
            print(f"{Color.GREEN}✅ 检测到校对稿已包含元数据，跳过元数据。")
        else:
            duration = self._get_audio_duration(audio_path)

            print(f"\n{Color.ORANGE}--- 请根据精准原文，输入最终的原文元数据 ---")
            if precision_text:
                print(f"{Color.ORANGE}精准原文参考：\n{precision_text[:500]}...")
            final_reference = input(f"{Color.DARK_PURPLE}请输入原文元数据（输完回车）：\n").strip()
            if final_reference:
                final_reference = f"> 原文：{final_reference}\n"
            print(f"{Color.ORANGE}-------------------------------------------")

            metadata_header = (
                f"> 标题：{base_name}\n"
                f"> 时间：{year}\n"
                f"> 时长：{duration}\n"
                f"> 作者：{author}\n"
                f"{final_reference}"
                "\n---\n\n"
            )

            final_content = metadata_header + current_written_text
            with open(final_output_filename, "w", encoding="utf-8") as f:
                f.write(final_content)
            print(f"{Color.GREEN}✅ 元数据已成功插入校对稿开头！")

        print(f"\n{Color.GREEN}===========================================")
        print("🎉 全部流程处理完成！")
        print(f"💡 最终文件列表：")
        print(f"  - 逐字稿: {transcript_filename.name}")
        print(f"  - 校对稿: {final_output_filename.name}")
        if precision_text:
            print(f"  - 精准原文: {precision_filename.name}")
        print(f"  - 检查报告: {verification_filename.name}")
        print(f"===========================================\n")


def run(*, folder_path, year, author, reference_text, asr_model_name,
        available_models, selected_model,
        text_system_prompt, text_user_prompt_template,
        verifier_system_prompt, verifier_user_prompt_template,
        precision_system_prompt="", precision_user_prompt_template=""):
    """执行完整流水线：文件发现 → 引擎初始化 → 运行"""

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

    print(f"{Color.GREEN}已自动识别任务：")
    print(f"  - 文件夹: {folder_path}")
    print(f"  - 音频文件: {audio_path_obj.name}")

    # 初始化引擎
    engine_cls, engine_kwargs = available_models[selected_model]

    print(f"{Color.DARK_PURPLE}正在初始化 Qwen3ASRFlashFiletrans 引擎...")
    asr_engine = Qwen3ASRFlashFiletrans(model_name=asr_model_name)

    print(f"{Color.DARK_PURPLE}正在初始化 {engine_cls.__name__} 引擎 ({selected_model} -> {engine_kwargs['model_name']})...")
    text_engine = engine_cls(**engine_kwargs)

    # 跑流水线
    pipeline = AudioProcessingPipeline(
        asr_engine=asr_engine,
        text_engine=text_engine
    )

    pipeline.run(
        audio_path=audio_path,
        fuzzy_reference_text=reference_text,
        text_sys_prompt=text_system_prompt,
        text_user_prompt_template=text_user_prompt_template,
        verifier_sys_prompt=verifier_system_prompt,
        verifier_user_prompt_template=verifier_user_prompt_template,
        precision_sys_prompt=precision_system_prompt,
        precision_user_prompt_template=precision_user_prompt_template,
        year=year,
        author=author,
        reference_text=reference_text
    )
