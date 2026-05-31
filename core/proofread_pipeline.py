"""
单次校对流水线（旧版，仅维护兼容）。
覆盖度检查 → 校对 → 信息丢失检查 → 提取原文 → 元数据
"""
import re
import shutil
import subprocess
import time
from pathlib import Path

from core.asr_engine import Qwen3ASRFlashFiletrans
from core.utils import Color


class AudioProcessingPipeline:
    """端到端音频处理流水线：转录 → 覆盖度检查 → 校对 → 信息丢失检查 → 定位并提取原文 → 元数据"""

    def __init__(self, asr_engine, text_engine, coverage_engine=None, verifier_engine=None, extract_engine=None):
        self.asr_engine = asr_engine
        self.text_engine = text_engine
        self._coverage_engine = coverage_engine or text_engine
        self._verifier_engine = verifier_engine or text_engine
        self._extract_engine = extract_engine or text_engine

    def _get_audio_duration(self, audio_path):
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(audio_path)],
            capture_output=True, text=True
        )
        duration_seconds = int(float(result.stdout.strip()))
        hours, remainder = divmod(duration_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _save_thinking(self, label: str, parent_dir: Path, engine=None):
        """保存思维链内容到文件（若引擎有返回）"""
        eng = engine or self.text_engine
        if hasattr(eng, '_last_reasoning_content') and eng._last_reasoning_content:
            thinking_path = parent_dir / f"思维链_{label}.md"
            thinking_path.write_text(eng._last_reasoning_content, encoding="utf-8")
            eng._last_reasoning_content = None
            print(f"{Color.GREEN}🧠 思维链已保存: {thinking_path.name}")

    @staticmethod
    def _format_time(seconds: float) -> str:
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
            coverage_sys_prompt="",
            coverage_user_prompt_template="",
            text_sys_prompt="",
            text_user_prompt_template="",
            verifier_sys_prompt="",
            verifier_user_prompt_template="",
            extract_sys_prompt="",
            extract_user_prompt_template="",
            example_text="",
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
        duration = self._get_audio_duration(audio_path)
        print(f"{Color.ORANGE}⏱️ 音频时长: {duration}，供参考选取原文范围{Color.END}")
        print(f"请在 {fuzzy_filename.name} 中整理模糊原文，保存后回到此处按回车继续...")
        input()
        fuzzy_reference_text = fuzzy_filename.read_text("utf-8").strip()
        if not fuzzy_reference_text:
            print(f"{Color.RED}错误：模糊原文为空，请填入内容后重新运行。")
            exit(1)

        # 步骤 1: 整段音频直接转录 (带缓存)
        print(f"\n[1/5] 开始处理整段音频转录...")
        if transcript_filename.exists():
            print(f"{Color.GREEN}✅ 检测到已存在逐字稿，直接复用: {transcript_filename.name}")
            with open(transcript_filename, "r", encoding="utf-8") as f:
                transcript_text = f.read()
        else:
            print(f"正在调用 ASR 引擎生成逐字稿...")
            _t0 = time.time()
            transcript_text = self.asr_engine.recognize(audio_path)
            _elapsed = time.time() - _t0
            _total_time += _elapsed
            print(f"{Color.GREEN}⏱️ ASR 转录耗时: {self._format_time(_elapsed)}")
            with open(transcript_filename, "w", encoding="utf-8") as f:
                f.write(transcript_text)
            print(f"{Color.GREEN}逐字稿已保存至: {transcript_filename}")

        # 步骤 2: 覆盖度检查
        coverage_cache = parent_dir / "覆盖度检查_结果.md"
        print(f"\n[2/6] 开始检查原文覆盖度...")
        if coverage_sys_prompt and coverage_user_prompt_template:
            if coverage_cache.exists():
                print(f"{Color.GREEN}✅ 检测到已存在覆盖度检查结果，直接复用: {coverage_cache.name}")
                coverage_result = coverage_cache.read_text("utf-8")
            else:
                coverage_prompt = coverage_user_prompt_template.format(
                    transcript_text=transcript_text,
                    fuzzy_reference_text=fuzzy_reference_text
                )
                _t0 = time.time()
                coverage_result = "".join(self._coverage_engine.generate_stream(coverage_sys_prompt, coverage_prompt))
                _elapsed = time.time() - _t0
                _total_time += _elapsed
                print(f"{Color.GREEN}⏱️ 覆盖度检查模型调用耗时: {self._format_time(_elapsed)}")
                self._save_thinking("覆盖度检查", parent_dir, self._coverage_engine)

            print(f"{Color.ORANGE}--- 覆盖度检查结果 ---")
            print(coverage_result)
            print(f"{Color.ORANGE}------------------------")

            if "【没有多讲】" not in coverage_result:
                print(f"{Color.RED}❌ 模糊原文未全面覆盖逐字稿内容，请补充模糊原文后重新运行。{Color.END}")
                exit(1)
            else:
                if coverage_sys_prompt and coverage_user_prompt_template and not coverage_cache.exists():
                    coverage_cache.write_text(coverage_result, "utf-8")
                print(f"{Color.GREEN}✅ 模糊原文已全面覆盖逐字稿内容")

        # 步骤 3+4: 校对 + 信息检查（共享重试，最多 3 次）
        _retry_left = 3
        _attempt = 0
        while _attempt < _retry_left:
            if _attempt > 0:
                for f in parent_dir.glob(f"{base_name}_校对稿_*.md"):
                    f.unlink()
                for f in parent_dir.glob("丢失信息检查_*.md"):
                    f.unlink()
                print(f"{Color.RED}{'=' * 50}")
                print(f"⏳ 第 {_attempt+1} 次重试...")
                print(f"{'=' * 50}{Color.END}")

            print(f"\n[3/6] 开始处理校对稿...")
            existing = list(parent_dir.glob(f"{base_name}_校对稿_*.md"))
            if existing:
                final_output_filename = existing[0]
                print(f"{Color.GREEN}✅ 检测到已存在校对稿，直接复用: {final_output_filename.name}")
                written_text = final_output_filename.read_text("utf-8")
            else:
                print(f"正在使用 {self.text_engine.model_name} 将逐字稿整理为校对稿...")
                user_prompt = text_user_prompt_template.format(
                    fuzzy_reference_text=fuzzy_reference_text,
                    transcript_text=transcript_text,
                    example_text=example_text,
                )
                _t0 = time.time()
                written_text = "".join(self.text_engine.generate_stream(text_sys_prompt, user_prompt))
                _elapsed = time.time() - _t0
                _total_time += _elapsed
                print(f"{Color.GREEN}⏱️ 校对稿模型调用耗时: {self._format_time(_elapsed)}")
                self._save_thinking("校对", parent_dir)
                if written_text.startswith("【校对稿】"):
                    written_text = written_text[len("【校对稿】"):].strip()
                final_output_filename.write_text(written_text, "utf-8")

            transcript_len = len(transcript_text)
            written_len = len(written_text)
            ratio = (written_len / transcript_len * 100) if transcript_len > 0 else 0.0
            print(f"{Color.ORANGE}📊 字数统计：校对稿 {written_len} / 逐字稿 {transcript_len} = {ratio:.1f}%")
            ratio_filename = parent_dir / f"{base_name}_校对稿_{ratio:.1f}%.md"
            if final_output_filename != ratio_filename and final_output_filename.exists():
                final_output_filename.rename(ratio_filename)
                final_output_filename = ratio_filename
                print(f"{Color.GREEN}校对稿已保存至: {final_output_filename.name}")

            if ratio < 50.0:
                print(f"{Color.ORANGE}⚠️ 浓缩率 {ratio:.1f}% 低于 50%，跳过信息丢失检查直接重试...{Color.END}")
                _attempt += 1
                if _attempt >= _retry_left:
                    print(f"{Color.RED}❌ 经 {_retry_left} 次尝试浓缩率仍低于 50%。流水线终止。{Color.END}")
                    exit(1)
                continue

            print(f"\n[4/6] 开始检查信息丢失情况...")
            cached = list(parent_dir.glob("丢失信息检查_*.md"))
            if cached:
                verification_filename = cached[0]
                print(f"{Color.GREEN}✅ 检测到已存在检查报告，直接复用: {verification_filename.name}")
                verification_report = verification_filename.read_text("utf-8")
            else:
                print(f"正在对比逐字稿与校对稿...")
                verifier_user_prompt = verifier_user_prompt_template.format(
                    transcript_text=transcript_text,
                    written_text=written_text,
                )
                _t0 = time.time()
                verification_report = "".join(self._verifier_engine.generate_stream(verifier_sys_prompt, verifier_user_prompt))
                _elapsed = time.time() - _t0
                _total_time += _elapsed
                print(f"{Color.GREEN}⏱️ 信息丢失检查模型调用耗时: {self._format_time(_elapsed)}")
                self._save_thinking("信息丢失检查", parent_dir, self._verifier_engine)
                has_loss = "【无遗漏信息】" not in verification_report
                tag = "有遗漏" if has_loss else "无遗漏"
                verification_filename = parent_dir / f"丢失信息检查_{tag}.md"
                verification_filename.write_text(verification_report, "utf-8")
                print(f"{Color.GREEN}检查报告已保存至: {verification_filename}（{tag}）")

            if "【无遗漏信息】" in verification_report:
                break

            print(f"{Color.ORANGE}⚠️ 检测到可能的信息遗漏。{Color.END}")
            print(f"请查看 {verification_filename.name}，判断遗漏是否可以接受。{Color.END}")
            user_choice = input(
                f"{Color.ORANGE}输入 y 接受本次校对稿，继续下一步；"
                f"输入 n 重试（将归档当前文件后重新生成）"
                f"[{Color.GREEN}y{Color.ORANGE}/{Color.RED}n{Color.ORANGE}]{Color.END} ").strip().lower()
            if user_choice == "y":
                print(f"{Color.GREEN}✅ 用户接受当前遗漏，继续下一步。{Color.END}")
                break

            print(f"{Color.ORANGE}📦 归档本次生成的文件后重试...{Color.END}")
            archive_base = parent_dir / "遗漏记录"
            archive_base.mkdir(parents=True, exist_ok=True)
            existing_nums = [int(d.name) for d in archive_base.iterdir() if d.is_dir() and d.name.isdigit()]
            next_num = max(existing_nums) + 1 if existing_nums else 1
            archive_dir = archive_base / str(next_num)
            archive_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(final_output_filename, archive_dir / final_output_filename.name)
            for f in parent_dir.glob("丢失信息检查_*.md"):
                shutil.copy2(f, archive_dir / f.name)
                f.unlink()
            print(f"{Color.GREEN}✅ 已归档至: {archive_dir}")
            _attempt += 1
            if _attempt >= _retry_left:
                print(f"{Color.RED}❌ 经 {_retry_left} 次尝试信息仍有遗漏。流水线终止。{Color.END}")
                exit(1)

        origin_txt = parent_dir / "原文.md"
        print(f"\n[5/6] 正在通过模型定位并提取纯原文...")
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
            extracted = "".join(self._extract_engine.generate_stream(extract_sys_prompt, extract_prompt)).strip()
            _elapsed = time.time() - _t0
            _total_time += _elapsed
            print(f"{Color.GREEN}⏱️ 定位提取原文模型调用耗时: {self._format_time(_elapsed)}")
            self._save_thinking("定位提取原文", parent_dir, self._extract_engine)
            extracted = extracted.replace("*", "").strip()
            extracted = extracted.replace("』", "").replace("『", "")
            if "\n" in extracted:
                has_newline = True
                extracted = extracted.replace("\n", "")
            else:
                has_newline = False
            origin_txt.write_text(extracted, "utf-8")
            print(f"{Color.GREEN}✅ 已通过模型提取原文到 {origin_txt.name}")
            self._print_head_tail(extracted)
            if has_newline:
                print(f"{Color.ORANGE}⚠️ 提取的原文中包含换行符，已自动移除。{Color.END}")
            else:
                print(f"{Color.GREEN}✅ 提取的原文中未包含换行符")
        else:
            origin_txt.write_text("", "utf-8")
            print(f"{Color.GREEN}已创建空的 {origin_txt}")

        print(f"{Color.GREEN}⏱️ 本次全部模型调用总耗时: {self._format_time(_total_time)}")

        print(f"\n[6/6] 准备补充元数据信息...")
        with open(final_output_filename, "r", encoding="utf-8") as f:
            current_written_text = f.read()
        if current_written_text.strip().startswith("> 标题："):
            print(f"{Color.GREEN}✅ 检测到校对稿已包含元数据，跳过元数据。")
        else:
            duration = self._get_audio_duration(audio_path)
            input(f"请在 {origin_txt.name} 中检查整理，保存后回到此处按回车继续...")
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
        print(f"===========================================\n")


def run(*, folder_path, year, author, reference_text, asr_model_name,
        available_models, selected_model,
        coverage_selected_model=None, verifier_selected_model=None, extract_selected_model=None,
        coverage_system_prompt="", coverage_user_prompt_template="",
        text_system_prompt, text_user_prompt_template,
        verifier_system_prompt, verifier_user_prompt_template,
        extract_system_prompt="", extract_user_prompt_template="",
        example_text=""):
    """执行完整流水线：文件发现 → 引擎初始化 → 运行"""

    if not folder_path.exists():
        print(f"{Color.RED}错误：找不到文件夹 {folder_path}")
        exit(1)

    audio_files = list(folder_path.glob("*.mp3"))
    if not audio_files:
        print(f"{Color.RED}错误：在 {folder_path} 下没找到 .mp3 文件")
        exit(1)
    audio_path_obj = next((f for f in audio_files if f.stem == folder_path.name), audio_files[0])
    audio_path = str(audio_path_obj)

    print(f"{Color.GREEN}已自动识别任务：")
    print(f"  - 文件夹: {folder_path}")
    print(f"  - 音频文件: {audio_path_obj.name}")

    def _init_engine(label, model_key):
        if model_key and model_key in available_models:
            cls, kwargs = available_models[model_key]
            print(f"正在初始化 {cls.__name__} 引擎 ({model_key} -> {kwargs['model_name']})...")
            return cls(**kwargs)
        print(f"{label}：未指定引擎，将复用主引擎")
        return None

    print(f"正在初始化 Qwen3ASRFlashFiletrans 引擎...")
    asr_engine = Qwen3ASRFlashFiletrans(model_name=asr_model_name)
    text_engine = _init_engine("校对稿", selected_model)
    coverage_engine = _init_engine("覆盖度检查", coverage_selected_model)
    verifier_engine = _init_engine("信息丢失检查", verifier_selected_model)
    extract_engine = _init_engine("定位提取原文", extract_selected_model)

    pipeline = AudioProcessingPipeline(
        asr_engine=asr_engine,
        text_engine=text_engine,
        coverage_engine=coverage_engine,
        verifier_engine=verifier_engine,
        extract_engine=extract_engine
    )
    pipeline.run(
        audio_path=audio_path,
        fuzzy_reference_text=reference_text,
        coverage_sys_prompt=coverage_system_prompt,
        coverage_user_prompt_template=coverage_user_prompt_template,
        text_sys_prompt=text_system_prompt,
        text_user_prompt_template=text_user_prompt_template,
        verifier_sys_prompt=verifier_system_prompt,
        verifier_user_prompt_template=verifier_user_prompt_template,
        extract_sys_prompt=extract_system_prompt,
        extract_user_prompt_template=extract_user_prompt_template,
        example_text=example_text,
        year=year,
        author=author,
        reference_text=reference_text
    )