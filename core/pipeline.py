import re
import time
from pathlib import Path
from pydub import AudioSegment

from core.asr_engine import Qwen3ASRFlashFiletrans
from core.utils import Color


class AudioProcessingPipeline:
    """端到端音频处理流水线：转录 → 校对 → 信息丢失检查 → 定位并提取原文 → 元数据"""

    def __init__(self, asr_engine, text_engine, fast_text_engine=None):
        self.asr_engine = asr_engine
        self.text_engine = text_engine
        self._fast_engine = fast_text_engine or text_engine

    def _get_audio_duration(self, audio_path):
        audio = AudioSegment.from_mp3(audio_path)
        duration_seconds = int(audio.duration_seconds)
        hours, remainder = divmod(duration_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _save_thinking(self, label: str, parent_dir: Path, engine=None):
        """保存思维链内容到文件（若引擎有返回）"""
        eng = engine or self.text_engine
        if hasattr(eng, '_last_reasoning_content') and eng._last_reasoning_content:
            thinking_path = parent_dir / f"思维链_{label}.md"
            thinking_path.write_text(eng._last_reasoning_content, encoding="utf-8")
            eng._last_reasoning_content = None  # 消费后清空
            print(f"{Color.GREEN}🧠 思维链已保存: {thinking_path.name}")

    @staticmethod
    def _format_time(seconds: float) -> str:
        """将秒数格式化为 xx分xx秒"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        if minutes > 0:
            return f"{minutes}分{secs}秒"
        return f"{secs}秒"

    @staticmethod
    def _print_head_tail(text, label="原文.md"):
        """打印文本的前两句和后两句"""
        sents = re.split(r'(?<=[。！？])', text)
        sents = [s.strip() for s in sents if s.strip()]
        if not sents:
            return
        if sents:
            print(f"{Color.ORANGE}📄 {label} 前两句：")
            print(f"  {''.join(sents[:2])}")
            if len(sents) > 2:
                print(f"{Color.ORANGE}📄 {label} 最后两句：")
                print(f"  {''.join(sents[-2:])}")

    def run(self,
            audio_path,
            fuzzy_reference_text="",
            text_sys_prompt="",
            text_user_prompt_template="",
            verifier_sys_prompt="",
            verifier_user_prompt_template="",
            extract_sys_prompt="",
            extract_user_prompt_template="",
            year="",
            author="",
            reference_text=""):

        audio_path_obj = Path(audio_path)
        base_name = audio_path_obj.stem
        parent_dir = audio_path_obj.parent
        _total_time = 0.0

        transcript_filename = parent_dir / f"{base_name}_逐字稿.md"
        final_output_filename = parent_dir / f"{base_name}_校对稿.md"
        fuzzy_filename = parent_dir / "模糊原文.md"

        # 模糊原文：文件优先，没有则创建后等待用户编辑
        if not fuzzy_filename.exists():
            with open(fuzzy_filename, "w", encoding="utf-8") as f:
                f.write(fuzzy_reference_text)
            print(f"{Color.GREEN}✅ 已创建 {fuzzy_filename}")
        else:
            print(f"{Color.GREEN}检测到已存在 {fuzzy_filename.name}")
        print(f"{Color.DARK_PURPLE}请在 {fuzzy_filename.name} 中整理模糊原文，保存后回到此处按回车继续...")
        input()
        fuzzy_reference_text = fuzzy_filename.read_text("utf-8").strip()
        if not fuzzy_reference_text:
            print(f"{Color.RED}错误：模糊原文为空，请填入内容后重新运行。")
            exit(1)

        # 步骤 1: 整段音频直接转录 (带缓存)
        print(f"\n{Color.DARK_PURPLE}[1/5] 开始处理整段音频转录...")
        if transcript_filename.exists():
            print(f"{Color.GREEN}✅ 检测到已存在逐字稿，直接复用: {transcript_filename.name}")
            with open(transcript_filename, "r", encoding="utf-8") as f:
                transcript_text = f.read()
        else:
            print(f"{Color.DARK_PURPLE}正在调用 ASR 引擎生成逐字稿...")
            _t0 = time.time()
            transcript_text = self.asr_engine.recognize(audio_path)
            _elapsed = time.time() - _t0
            _total_time += _elapsed
            print(f"{Color.GREEN}⏱️ ASR 转录耗时: {self._format_time(_elapsed)}")

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

            # 调用模型 + 自动重试（当浓缩率异常低时）
            min_ratio = 0.50
            for retry in range(3):
                _t0 = time.time()
                written_text = self.text_engine.generate(text_sys_prompt, user_prompt)
                _elapsed = time.time() - _t0
                _total_time += _elapsed
                print(f"{Color.GREEN}⏱️ 校对稿模型调用耗时: {self._format_time(_elapsed)}")
                self._save_thinking("校对", parent_dir)

                ratio = len(written_text) / len(transcript_text) if transcript_text else 1.0
                if ratio >= min_ratio:
                    break
                print(f"{Color.ORANGE}⚠️ 浓缩率 {ratio:.1%} 低于 {min_ratio:.0%}，正在重试 ({retry+1}/3)...{Color.END}")
                # 重试时在 prompt 末尾追加约束
                user_prompt += "\n\n【重要】上次输出的内容太短了，丢失了大量信息，这次请务必完整输出原文的全部内容，不要压缩。"

            with open(final_output_filename, "w", encoding="utf-8") as f:
                f.write(written_text)
            print(f"{Color.GREEN}校对稿已保存至: {final_output_filename}")

        # 步骤 3: 信息丢失检查 (带缓存)
        print(f"\n{Color.DARK_PURPLE}[3/5] 开始检查信息丢失情况...")

        transcript_len = len(transcript_text)
        written_len = len(written_text)
        ratio = (written_len / transcript_len * 100) if transcript_len > 0 else 0.0
        print(f"{Color.ORANGE}📊 字数统计：校对稿 {written_len} / 逐字稿 {transcript_len} = {ratio:.1f}%")

        cached = list(parent_dir.glob(f"丢失信息检查_*.md"))
        if cached:
            verification_filename = cached[0]
            print(f"{Color.GREEN}✅ 检测到已存在检查报告，直接复用: {verification_filename.name}")
            verification_report = verification_filename.read_text("utf-8")
        else:
            print(f"{Color.DARK_PURPLE}正在对比逐字稿与校对稿，检查是否有关键信息丢失...")
            verifier_user_prompt = verifier_user_prompt_template.format(
                transcript_text=transcript_text,
                written_text=written_text,
            )

            _t0 = time.time()
            verification_report = self.text_engine.generate(verifier_sys_prompt, verifier_user_prompt)
            _elapsed = time.time() - _t0
            _total_time += _elapsed
            print(f"{Color.GREEN}⏱️ 信息丢失检查模型调用耗时: {self._format_time(_elapsed)}")
            self._save_thinking("信息丢失检查", parent_dir)
            has_loss = "【无遗漏信息】" not in verification_report
            tag = "有遗漏" if has_loss else "无遗漏"
            verification_filename = parent_dir / f"丢失信息检查_{tag}.md"
            verification_filename.write_text(verification_report, "utf-8")
            print(f"{Color.GREEN}检查报告已保存至: {verification_filename}（{tag}）")

        print(f"{Color.ORANGE}--- 信息丢失检查报告摘要 ---")
        print("\n".join(verification_report.split("\n")[:10]))
        if len(verification_report.split("\n")) > 10:
            print("...")
        print(f"{Color.ORANGE}----------------------------")

        # 有遗漏信息则终止流水线
        if "【无遗漏信息】" not in verification_report:
            print(f"{Color.RED}❌ 检测到信息遗漏，流水线终止。请手动修改校对稿后重新运行。{Color.END}")
            exit(1)

        # 步骤 4: 精准定位并提取原文 (带缓存)
        origin_txt = parent_dir / "原文.md"
        print(f"\n{Color.DARK_PURPLE}[4/5] 正在通过模型定位并提取纯原文...")
        if origin_txt.exists():
            print(f"{Color.GREEN}✅ 检测到已存在 {origin_txt.name}，直接复用")
            content_for_print = origin_txt.read_text("utf-8").strip()
            if content_for_print:
                self._print_head_tail(content_for_print)
        elif extract_sys_prompt and extract_user_prompt_template:
            extract_prompt = extract_user_prompt_template.format(
                written_text=written_text,
                fuzzy_reference_text=fuzzy_reference_text
            )
            _t0 = time.time()
            extracted = self._fast_engine.generate(extract_sys_prompt, extract_prompt).strip()
            _elapsed = time.time() - _t0
            _total_time += _elapsed
            print(f"{Color.GREEN}⏱️ 定位提取原文模型调用耗时: {self._format_time(_elapsed)}")
            self._save_thinking("定位提取原文", parent_dir, self._fast_engine)
            origin_txt.write_text(extracted, "utf-8")
            print(f"{Color.GREEN}✅ 已通过模型提取原文到 {origin_txt.name}")
            self._print_head_tail(extracted)
        else:
            origin_txt.write_text("", "utf-8")
            print(f"{Color.GREEN}已创建空的 {origin_txt}")

        print(f"{Color.GREEN}⏱️ 本次全部模型调用总耗时: {self._format_time(_total_time)}")

        # 步骤 5: 补充元数据并插入校对稿
        print(f"\n{Color.DARK_PURPLE}[5/5] 准备补充元数据信息...")
        with open(final_output_filename, "r", encoding="utf-8") as f:
            current_written_text = f.read()

        if current_written_text.strip().startswith("> 标题："):
            print(f"{Color.GREEN}✅ 检测到校对稿已包含元数据，跳过元数据。")
        else:
            duration = self._get_audio_duration(audio_path)

            input(f"{Color.DARK_PURPLE}请在 {origin_txt.name} 中检查整理，保存后回到此处按回车继续...")
            final_reference = origin_txt.read_text("utf-8").strip()
            final_reference = f"> 原文：{final_reference}\n" if final_reference else ""
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
        print(f"  - 检查报告: {verification_filename.name}")
        print(f"===========================================\n")


class BatchTranscriptionPipeline:
    """批量转录流水线：递归扫描文件夹，对每个音频文件调用 ASR 引擎生成逐字稿。"""

    def __init__(self, asr_engine, audio_extensions=None):
        self.asr_engine = asr_engine
        self.audio_extensions = audio_extensions or {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".wma"}

    def run(self, folder_path: str | Path):
        folder_path = Path(folder_path)
        audio_files = sorted(
            [f for f in folder_path.rglob("*") if f.is_file() and f.suffix.lower() in self.audio_extensions],
            key=lambda p: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(p))]
        )
        if not audio_files:
            print(f"{Color.RED}错误：在 {folder_path} 下没找到音频文件")
            exit(1)

        print(f"{Color.GREEN}找到 {len(audio_files)} 个音频文件")
        print(f"{Color.DARK_PURPLE}使用 {type(self.asr_engine).__name__} ASR 引擎")

        completed = 0
        total_seconds = 0.0

        for audio_path in audio_files:
            transcript_path = audio_path.parent / f"{audio_path.stem}_逐字稿.md"
            if transcript_path.exists():
                print(f"{Color.GREEN}[{audio_path.stem}] 逐字稿已存在，跳过")
                continue

            print(f"{Color.DARK_PURPLE}[{audio_path.stem}] 正在转录...")
            transcript_text = self.asr_engine.recognize(str(audio_path.resolve()))
            transcript_path.write_text(transcript_text, "utf-8")
            print(f"{Color.GREEN}[{audio_path.stem}] 逐字稿已保存")

            completed += 1
            duration_sec = len(AudioSegment.from_file(str(audio_path))) / 1000.0
            total_seconds += duration_sec
            total_minutes = total_seconds / 60.0
            print(f"{Color.GREEN}✅ 已完成第 {completed} 条音频转录（累计 {total_minutes:.1f} 分钟）")

        print(f"\n{Color.GREEN}全部完成。")


def run(*, folder_path, year, author, reference_text, asr_model_name,
        available_models, selected_model, fast_selected_model=None,
        text_system_prompt, text_user_prompt_template,
        verifier_system_prompt, verifier_user_prompt_template,
        extract_system_prompt="", extract_user_prompt_template=""):
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

    # 初始化快速引擎（用于后两步，若不指定则回退到主引擎）
    fast_text_engine = None
    if fast_selected_model and fast_selected_model in available_models:
        fast_engine_cls, fast_engine_kwargs = available_models[fast_selected_model]
        print(f"{Color.DARK_PURPLE}正在初始化 {fast_engine_cls.__name__} 快速引擎 ({fast_selected_model} -> {fast_engine_kwargs['model_name']})...")
        fast_text_engine = fast_engine_cls(**fast_engine_kwargs)
    else:
        print(f"{Color.DARK_PURPLE}未指定快速引擎，后两步将复用主引擎")

    # 跑流水线
    pipeline = AudioProcessingPipeline(
        asr_engine=asr_engine,
        text_engine=text_engine,
        fast_text_engine=fast_text_engine
    )

    pipeline.run(
        audio_path=audio_path,
        fuzzy_reference_text=reference_text,
        text_sys_prompt=text_system_prompt,
        text_user_prompt_template=text_user_prompt_template,
        verifier_sys_prompt=verifier_system_prompt,
        verifier_user_prompt_template=verifier_user_prompt_template,
        extract_sys_prompt=extract_system_prompt,
        extract_user_prompt_template=extract_user_prompt_template,
        year=year,
        author=author,
        reference_text=reference_text
    )
