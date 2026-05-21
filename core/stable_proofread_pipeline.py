"""
稳扎版校对流水线：三步独立，每步单一职责。

流程：
① ASR逐字稿 → 初次校对稿
② 初次校对稿 + 模糊原文 → 精准原文
③ 精准原文 + ASR逐字稿 → 最终校对稿
"""
import re
import subprocess
import time
from pathlib import Path

from core.utils import Color


class TwoPassProofreadPipeline:
    """稳扎版校对流水线。一步调用，内部完成文件发现、引擎初始化、三步执行。"""

    def run(self,
            folder_path,
            year,
            author,
            asr_model_name,
            available_models,
            draft_model_key,
            refine_model_key,
            final_model_key,
            draft_sys_prompt="",
            draft_user_prompt_template="",
            refine_sys_prompt="",
            refine_user_prompt_template="",
            final_sys_prompt="",
            final_user_prompt_template="",
            refine_example_text="",
            example_text="",
            debug=False):

        # ── 初始化引擎 ─────────────────────
        from core.asr_engine import Qwen3ASRFlashFiletrans

        def _init(label, key):
            cls, kwargs = available_models[key]
            print(f"正在初始化 {cls.__name__} 引擎 ({key} -> {kwargs['model_name']})...")
            return cls(**kwargs)

        if not folder_path.exists():
            print(f"错误：找不到文件夹 {folder_path}")
            exit(1)

        audio_files = list(folder_path.glob("*.mp3"))
        if not audio_files:
            print(f"错误：在 {folder_path} 下没找到 .mp3 文件")
            exit(1)
        audio_path_obj = next((f for f in audio_files if f.stem == folder_path.name), audio_files[0])
        audio_path = str(audio_path_obj)

        print(f"已自动识别任务：")
        print(f"  - 文件夹: {folder_path}")
        print(f"  - 音频文件: {audio_path_obj.name}")

        print("正在初始化 Qwen3ASRFlashFiletrans 引擎...")
        asr_engine = Qwen3ASRFlashFiletrans(model_name=asr_model_name)
        draft_engine = _init("① 初次校对", draft_model_key)
        refine_engine = _init("② 精准原文", refine_model_key)
        final_engine = _init("③ 最终校对", final_model_key)

        # ── 开始执行 ───────────────────────
        audio_path_obj = Path(audio_path)
        base_name = audio_path_obj.stem
        parent_dir = audio_path_obj.parent
        _total_time = 0.0

        draft_filename = parent_dir / f"{base_name}_初次校对稿.md"
        precision_filename = parent_dir / f"{base_name}_精准原文.md"
        final_filename = parent_dir / f"{base_name}_最终校对稿.md"
        transcript_filename = parent_dir / f"{base_name}_逐字稿.md"
        fuzzy_filename = parent_dir / "模糊原文.md"

        # ── 前置：模糊原文 ──────────────────
        if not fuzzy_filename.exists():
            fuzzy_filename.write_text("", encoding="utf-8")
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

        # ── 步骤 0：ASR 转录 ────────────────
        print(f"\n[0/4] 开始处理整段音频转录...")
        if transcript_filename.exists():
            print(f"{Color.GREEN}✅ 检测到已存在逐字稿，直接复用: {transcript_filename.name}")
            transcript_text = transcript_filename.read_text("utf-8")
        else:
            self._confirm_step("ASR 转录", debug)
            print(f"正在调用 ASR 引擎生成逐字稿...")
            _t0 = time.time()
            transcript_text = asr_engine.recognize(audio_path)
            _elapsed = time.time() - _t0
            _total_time += _elapsed
            print(f"{Color.GREEN}⏱️ ASR 转录耗时: {self._format_time(_elapsed)}")
            transcript_filename.write_text(transcript_text, encoding="utf-8")
            print(f"{Color.GREEN}逐字稿已保存至: {transcript_filename}")

        # ── 步骤 1：初次校对稿 ──────────────
        print(f"\n[1/4] 正在生成初次校对稿（不参考原文）...")
        if draft_filename.exists():
            print(f"{Color.GREEN}✅ 检测到已存在初次校对稿，直接复用: {draft_filename.name}")
            draft_text = draft_filename.read_text("utf-8")
        else:
            self._confirm_step("初次校对（大模型）", debug)
            draft_prompt = draft_user_prompt_template.format(
                transcript_text=transcript_text,
                example_text=example_text,
            )
            _t0 = time.time()
            draft_text = draft_engine.generate(draft_sys_prompt, draft_prompt)
            _elapsed = time.time() - _t0
            _total_time += _elapsed
            print(f"{Color.GREEN}⏱️ 初次校对稿模型调用耗时: {self._format_time(_elapsed)}")
            self._save_thinking("初次校对", parent_dir, draft_engine)
            if draft_text.startswith("【校对稿】"):
                draft_text = draft_text[len("【校对稿】"):].strip()
                print(f"{Color.GREEN}✅ 后处理：已去掉【校对稿】前缀{Color.END}")
            draft_filename.write_text(draft_text, encoding="utf-8")
            print(f"{Color.GREEN}初次校对稿已保存至: {draft_filename.name}")

        draft_ratio = len(draft_text) / len(transcript_text) * 100
        print(f"{Color.ORANGE}📊 字数统计：初次校对稿 {len(draft_text)} / 逐字稿 {len(transcript_text)} = {draft_ratio:.1f}%{Color.END}")

        # ── 步骤 2：精准原文 ────────────────
        print(f"\n[2/4] 正在从模糊原文中提取精准原文...")
        if precision_filename.exists():
            print(f"{Color.GREEN}✅ 检测到已存在精准原文，直接复用: {precision_filename.name}")
            precision_text = precision_filename.read_text("utf-8")
        else:
            self._confirm_step("精准原文提取（大模型）", debug)
            refine_prompt = refine_user_prompt_template.format(
                refine_example_text=refine_example_text,
                draft_text=draft_text,
                fuzzy_reference_text=fuzzy_reference_text,
            )
            _t0 = time.time()
            precision_text = refine_engine.generate(refine_sys_prompt, refine_prompt)
            _elapsed = time.time() - _t0
            _total_time += _elapsed
            print(f"{Color.GREEN}⏱️ 精准原文提取耗时: {self._format_time(_elapsed)}")
            self._save_thinking("精准原文", parent_dir, refine_engine)
            precision_filename.write_text(precision_text, encoding="utf-8")
            print(f"{Color.GREEN}精准原文已保存至: {precision_filename.name}")

        # ── 提取 ** 包裹的摩诃止观正文 ──────
        main_body = "".join(re.findall(r"\*\*(.*?)\*\*", precision_text))

        # ── 步骤 3：最终校对稿 ──────────────
        print(f"\n[3/4] 正在使用精准原文生成最终校对稿...")
        if final_filename.exists():
            print(f"{Color.GREEN}✅ 检测到已存在校对稿，直接复用: {final_filename.name}")
            final_text = final_filename.read_text("utf-8")
        else:
            _final_retry = 3
            for _attempt in range(_final_retry):
                self._confirm_step("最终校对（大模型）", debug)
                final_prompt = final_user_prompt_template.format(
                    transcript_text=transcript_text,
                    precision_text=precision_text,
                    example_text=example_text,
                )
                _t0 = time.time()
                final_text = final_engine.generate(final_sys_prompt, final_prompt)
                _elapsed = time.time() - _t0
                _total_time += _elapsed
                print(f"{Color.GREEN}⏱️ 最终校对稿模型调用耗时: {self._format_time(_elapsed)}")
                self._save_thinking("最终校对", parent_dir, final_engine)
                if final_text.startswith("【校对稿】"):
                    final_text = final_text[len("【校对稿】"):].strip()
                    print(f"{Color.GREEN}✅ 后处理：已去掉【校对稿】前缀{Color.END}")

                # 浓缩率检查
                _ratio = len(final_text) / len(transcript_text) * 100
                print(f"{Color.ORANGE}📊 字数统计：校对稿 {len(final_text)} / 逐字稿 {len(transcript_text)} = {_ratio:.1f}%")
                if _ratio >= 50.0:
                    break
                if _attempt < _final_retry - 1:
                    print(f"{Color.ORANGE}⚠️ 浓缩率 {_ratio:.1f}% 低于 50%，第 {_attempt + 2} 次重试...{Color.END}")

            final_filename.write_text(final_text, encoding="utf-8")
            print(f"{Color.GREEN}最终校对稿已保存至: {final_filename.name}")

        # ── 浓缩率统计 ─────────────────────
        final_ratio = len(final_text) / len(transcript_text) * 100
        print(f"{Color.ORANGE}📊 字数统计：校对稿 {len(final_text)} / 逐字稿 {len(transcript_text)} = {final_ratio:.1f}%")

        print(f"{Color.GREEN}⏱️ 本次全部模型调用总耗时: {self._format_time(_total_time)}")

        # ── 补充元数据 ─────────────────────
        print(f"\n[4/4] 准备补充元数据信息...")
        current_text = final_filename.read_text("utf-8")
        if current_text.strip().startswith("> 标题："):
            print(f"{Color.GREEN}✅ 检测到校对稿已包含元数据，跳过。")
        else:
            duration = self._get_audio_duration(audio_path)
            main_body_header = f"> 摩诃止观正文：{main_body}\n" if main_body else ""
            metadata_header = (
                f"> 标题：{base_name}\n"
                f"> 时间：{year}\n"
                f"> 时长：{duration}\n"
                f"> 作者：{author}\n"
                f"{main_body_header}"
                "\n---\n\n"
            )
            final_filename.write_text(metadata_header + current_text, encoding="utf-8")
            print(f"{Color.GREEN}✅ 元数据已成功插入校对稿开头！{Color.END}")
            if main_body:
                print(f"{Color.ORANGE}📄 以下为插入的摩诃止观正文：{Color.END}")
                print(main_body)

        print(f"\n{Color.GREEN}===========================================")
        print("🎉 全部流程处理完成！")
        print(f"💡 最终文件列表：")
        print(f"  - 逐字稿: {transcript_filename.name}")
        print(f"  - 初次校对稿: {draft_filename.name}")
        print(f"  - 精准原文: {precision_filename.name}")
        print(f"  - 最终校对稿: {final_filename.name}")
        print(f"===========================================\n")

    # ── 工具方法 ─────────────────────────────

    @staticmethod
    def _confirm_step(label, debug):
        """调试模式下：询问用户是否继续"""
        if not debug:
            return
        choice = input(f"{Color.ORANGE}🔍 调试模式：即将执行【{label}】，输入 y 继续(n 退出): {Color.END}").strip().lower()
        if choice != "y":
            print(f"{Color.RED}❌ 用户终止。{Color.END}")
            exit(0)
        print(f"{Color.GREEN}✅ 继续执行【{label}】...{Color.END}")

    @staticmethod
    def _get_audio_duration(audio_path):
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(audio_path)],
            capture_output=True, text=True
        )
        duration_seconds = int(float(result.stdout.strip()))
        hours, remainder = divmod(duration_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    @staticmethod
    def _save_thinking(label, parent_dir, engine):
        if hasattr(engine, '_last_reasoning_content') and engine._last_reasoning_content:
            thinking_path = parent_dir / f"思维链_{label}.md"
            thinking_path.write_text(engine._last_reasoning_content, encoding="utf-8")
            engine._last_reasoning_content = None
            print(f"{Color.GREEN}🧠 思维链已保存: {thinking_path.name}")

    @staticmethod
    def _format_time(seconds):
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        if minutes > 0:
            return f"{minutes}分{secs}秒"
        return f"{secs}秒"
