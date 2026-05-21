"""
稳扎版校对流水线：四步独立，每步单一职责。

流程：
① ASR逐字稿 → 初次校对稿（不开思考，不碰原文）
② 初次校对稿 + 模糊原文 → 精准原文（开思考，砍掉多余）
③ 精准原文 + ASR逐字稿 → 最终校对稿（不开思考，内嵌自查遗漏）
④ 最终校对稿 + 精准原文 → 摩诃止观正文（开思考，高精度提取）
"""

import re
import shutil
import subprocess
import time
from pathlib import Path
from pydub import AudioSegment

from core.utils import Color


class TwoPassProofreadPipeline:
    """稳扎打校对流水线，四步各司其职。"""

    def __init__(self, asr_engine, draft_engine, refine_engine, final_engine, extract_engine):
        self.asr_engine = asr_engine
        self.draft_engine = draft_engine
        self.refine_engine = refine_engine
        self.final_engine = final_engine
        self.extract_engine = extract_engine

    # ── 工具方法 ─────────────────────────────────

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

    def _save_thinking(self, label: str, parent_dir: Path, engine):
        """保存思维链内容到文件（若引擎有返回）"""
        if hasattr(engine, '_last_reasoning_content') and engine._last_reasoning_content:
            thinking_path = parent_dir / f"思维链_{label}.md"
            thinking_path.write_text(engine._last_reasoning_content, encoding="utf-8")
            engine._last_reasoning_content = None
            print(f"{Color.GREEN}🧠 思维链已保存: {thinking_path.name}")

    @staticmethod
    def _format_time(seconds: float) -> str:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        if minutes > 0:
            return f"{minutes}分{secs}秒"
        return f"{secs}秒"

    @staticmethod
    def _print_head_tail(text, label="原文"):
        """打印文本的前两句和后两句"""
        sents = re.split(r'(?<=[。！？])', text)
        sents = [s.strip() for s in sents if s.strip()]
        if not sents:
            return
        print(f"{Color.ORANGE}📄 {label} 前两句：")
        print(f"  {''.join(sents[:2])}")
        if len(sents) > 2:
            print(f"{Color.ORANGE}📄 {label} 最后两句：")
            print(f"  {''.join(sents[-2:])}")

    # ── 主流程 ───────────────────────────────────

    def run(self,
            audio_path,
            fuzzy_reference_text="",
            draft_sys_prompt="",
            draft_user_prompt_template="",
            refine_sys_prompt="",
            refine_user_prompt_template="",
            final_sys_prompt="",
            final_user_prompt_template="",
            extract_sys_prompt="",
            extract_user_prompt_template="",
            refine_example_text="",
            example_text="",
            year="",
            author=""):

        audio_path_obj = Path(audio_path)
        base_name = audio_path_obj.stem
        parent_dir = audio_path_obj.parent
        _total_time = 0.0

        draft_filename = parent_dir / f"{base_name}_初次校对稿.md"
        precision_filename = parent_dir / f"{base_name}_精准原文.md"
        final_filename = parent_dir / f"{base_name}_校对稿.md"
        transcript_filename = parent_dir / f"{base_name}_逐字稿.md"
        origin_txt = parent_dir / "原文.txt"
        fuzzy_filename = parent_dir / "模糊原文.md"

        # ── 前置：模糊原文 ──────────────────────
        if not fuzzy_filename.exists():
            fuzzy_filename.write_text(fuzzy_reference_text, encoding="utf-8")
            print(f"{Color.GREEN}✅ 已创建 {fuzzy_filename}")
        else:
            print(f"{Color.GREEN}检测到已存在 {fuzzy_filename.name}")
        duration = self._get_audio_duration(audio_path)
        print(f"{Color.ORANGE}⏱️ 音频时长: {duration}，供参考选取原文范围{Color.END}")
        print(f"{Color.DARK_PURPLE}请在 {fuzzy_filename.name} 中整理模糊原文，保存后回到此处按回车继续...")
        input()
        fuzzy_reference_text = fuzzy_filename.read_text("utf-8").strip()
        if not fuzzy_reference_text:
            print(f"{Color.RED}错误：模糊原文为空，请填入内容后重新运行。")
            exit(1)

        # ── 步骤 0：ASR 转录（带缓存）───────────
        print(f"\n{Color.DARK_PURPLE}[0/5] 开始处理整段音频转录...")
        if transcript_filename.exists():
            print(f"{Color.GREEN}✅ 检测到已存在逐字稿，直接复用: {transcript_filename.name}")
            transcript_text = transcript_filename.read_text("utf-8")
        else:
            print(f"{Color.DARK_PURPLE}正在调用 ASR 引擎生成逐字稿...")
            _t0 = time.time()
            transcript_text = self.asr_engine.recognize(audio_path)
            _elapsed = time.time() - _t0
            _total_time += _elapsed
            print(f"{Color.GREEN}⏱️ ASR 转录耗时: {self._format_time(_elapsed)}")
            transcript_filename.write_text(transcript_text, encoding="utf-8")
            print(f"{Color.GREEN}逐字稿已保存至: {transcript_filename}")

        # ── 步骤 1：初次校对稿（不开思考，不碰原文）──
        print(f"\n{Color.DARK_PURPLE}[1/5] 正在生成初次校对稿（不参考原文）...")
        if draft_filename.exists():
            print(f"{Color.GREEN}✅ 检测到已存在初次校对稿，直接复用: {draft_filename.name}")
            draft_text = draft_filename.read_text("utf-8")
        else:
            draft_prompt = draft_user_prompt_template.format(
                transcript_text=transcript_text,
                example_text=example_text,
            )
            _t0 = time.time()
            draft_text = self.draft_engine.generate(draft_sys_prompt, draft_prompt)
            _elapsed = time.time() - _t0
            _total_time += _elapsed
            print(f"{Color.GREEN}⏱️ 初次校对稿模型调用耗时: {self._format_time(_elapsed)}")
            self._save_thinking("初次校对", parent_dir, self.draft_engine)

            # 后处理：去掉模型擅加的前缀
            if draft_text.startswith("【校对稿】"):
                draft_text = draft_text[len("【校对稿】"):].strip()

            draft_filename.write_text(draft_text, encoding="utf-8")
            print(f"{Color.GREEN}初次校对稿已保存至: {draft_filename.name}")

        # 初次校对浓缩率
        draft_ratio = len(draft_text) / len(transcript_text) * 100
        print(f"{Color.ORANGE}📊 字数统计：初次校对稿 {len(draft_text)} / 逐字稿 {len(transcript_text)} = {draft_ratio:.1f}%{Color.END}")

        # ── 步骤 2：精准原文（开思考，砍掉多余）────
        print(f"\n{Color.DARK_PURPLE}[2/5] 正在从模糊原文中提取精准原文...")
        if precision_filename.exists():
            print(f"{Color.GREEN}✅ 检测到已存在精准原文，直接复用: {precision_filename.name}")
            precision_text = precision_filename.read_text("utf-8")
        else:
            refine_prompt = refine_user_prompt_template.format(
                refine_example_text=refine_example_text,
                draft_text=draft_text,
                fuzzy_reference_text=fuzzy_reference_text,
            )
            _t0 = time.time()
            precision_text = self.refine_engine.generate(refine_sys_prompt, refine_prompt)
            _elapsed = time.time() - _t0
            _total_time += _elapsed
            print(f"{Color.GREEN}⏱️ 精准原文提取耗时: {self._format_time(_elapsed)}")
            self._save_thinking("精准原文", parent_dir, self.refine_engine)

            # 后处理：去掉 **、『』、换行符
            precision_text = precision_text.replace("*", "").replace("『", "").replace("』", "").replace("\n", "").strip()

            precision_filename.write_text(precision_text, encoding="utf-8")
            print(f"{Color.GREEN}精准原文已保存至: {precision_filename.name}")

        # ── 步骤 3：最终校对稿（不开思考，内嵌自查）──
        print(f"\n{Color.DARK_PURPLE}[3/5] 正在使用精准原文生成最终校对稿...")
        if final_filename.exists():
            print(f"{Color.GREEN}✅ 检测到已存在校对稿，直接复用: {final_filename.name}")
            final_text = final_filename.read_text("utf-8")
        else:
            final_prompt = final_user_prompt_template.format(
                transcript_text=transcript_text,
                precision_text=precision_text,
                example_text=example_text,
            )
            _t0 = time.time()
            final_text = self.final_engine.generate(final_sys_prompt, final_prompt)
            _elapsed = time.time() - _t0
            _total_time += _elapsed
            print(f"{Color.GREEN}⏱️ 最终校对稿模型调用耗时: {self._format_time(_elapsed)}")
            self._save_thinking("最终校对", parent_dir, self.final_engine)

            # 后处理：去掉模型擅加的前缀
            if final_text.startswith("【校对稿】"):
                final_text = final_text[len("【校对稿】"):].strip()

            final_filename.write_text(final_text, encoding="utf-8")
            print(f"{Color.GREEN}最终校对稿已保存至: {final_filename.name}")

        # ── 浓缩率统计 ─────────────────────────
        transcript_len = len(transcript_text)
        final_len = len(final_text)
        ratio = (final_len / transcript_len * 100) if transcript_len > 0 else 0.0
        print(f"{Color.ORANGE}📊 字数统计：校对稿 {final_len} / 逐字稿 {transcript_len} = {ratio:.1f}%")

        # ── 步骤 4：提取正文（开思考，高精度）────
        print(f"\n{Color.DARK_PURPLE}[4/5] 正在通过模型提取摩诃止观正文...")
        if origin_txt.exists():
            print(f"{Color.GREEN}✅ 检测到已存在 {origin_txt.name}，直接复用")
            content_for_print = origin_txt.read_text("utf-8").strip()
            if content_for_print:
                self._print_head_tail(content_for_print)
        else:
            extract_prompt = extract_user_prompt_template.format(
                written_text=final_text,
                precision_text=precision_text,
            )
            _t0 = time.time()
            extracted = self.extract_engine.generate(extract_sys_prompt, extract_prompt).strip()
            _elapsed = time.time() - _t0
            _total_time += _elapsed
            print(f"{Color.GREEN}⏱️ 正文提取耗时: {self._format_time(_elapsed)}")
            self._save_thinking("定位提取原文", parent_dir, self.extract_engine)

            # 后处理
            extracted = extracted.replace("*", "").replace("『", "").replace("』", "").replace("\n", "").strip()

            # 去掉模型擅加的前缀
            if extracted.startswith("【校对稿】"):
                extracted = extracted[len("【校对稿】"):].strip()

            origin_txt.write_text(extracted, encoding="utf-8")
            print(f"{Color.GREEN}✅ 已通过模型提取原文到 {origin_txt.name}")
            self._print_head_tail(extracted)

        print(f"{Color.GREEN}⏱️ 本次全部模型调用总耗时: {self._format_time(_total_time)}")

        # ── 步骤 5：补充元数据 ──────────────────
        print(f"\n{Color.DARK_PURPLE}[5/5] 准备补充元数据信息...")
        current_text = final_filename.read_text("utf-8")

        if current_text.strip().startswith("> 标题："):
            print(f"{Color.GREEN}✅ 检测到校对稿已包含元数据，跳过。")
        else:
            duration = self._get_audio_duration(audio_path)
            input(f"{Color.DARK_PURPLE}请在 {origin_txt.name} 中检查整理，保存后回到此处按回车继续...")
            final_reference = origin_txt.read_text("utf-8").strip()
            final_reference_line = f"> 原文：{final_reference}\n" if final_reference else ""

            metadata_header = (
                f"> 标题：{base_name}\n"
                f"> 时间：{year}\n"
                f"> 时长：{duration}\n"
                f"> 作者：{author}\n"
                f"{final_reference_line}"
                "\n---\n\n"
            )

            final_content = metadata_header + current_text
            final_filename.write_text(final_content, encoding="utf-8")
            print(f"{Color.GREEN}✅ 元数据已成功插入校对稿开头！")

        print(f"\n{Color.GREEN}===========================================")
        print("🎉 全部流程处理完成！")
        print(f"💡 最终文件列表：")
        print(f"  - 逐字稿: {transcript_filename.name}")
        print(f"  - 初次校对稿: {draft_filename.name}")
        print(f"  - 精准原文: {precision_filename.name}")
        print(f"  - 校对稿: {final_filename.name}")
        print(f"  - 摩诃止观正文: {origin_txt.name}")
        print(f"===========================================\n")
