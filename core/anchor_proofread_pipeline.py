"""
编号锚点校对流水线（完整版）。
融合精准末尾定位 + 编号锚点校对 + 遗漏检查 + 脚本去编号。

流程：
  [0/8] ASR 转录
  [1/8] 模糊原文
  [2/8] 脚注重排
  [3/8] 末尾定位
  [4/8] Python 截断得精准原文
  [5/8] 编号锚点校对
  [6/8] 遗漏检查
  [7/8] 脚本去编号
  [8/8] 插入元数据
"""

import re
import time
from pathlib import Path
from typing import Any

from core.utils import Color, format_duration
from core.asr_engine import Qwen3ASRFlashFiletrans

from .pipeline_steps.step_asr import step_asr
from .pipeline_steps.step_fuzzy import step_fuzzy_text
from .pipeline_steps.step_rearrange import step_rearrange
from .pipeline_steps.step_locate import step_locate_and_precision
from .pipeline_steps.step_anchor import step_anchor_proofread
from .pipeline_steps.step_miss_check import step_miss_check
from .pipeline_steps.step_remove_numbers import step_remove_numbers
from .pipeline_steps.step_finalize import step_finalize


class AnchorProofreadPipeline:
    """编号锚点校对流水线。融合精准末端定位 + 编号锚点防遗漏机制。"""

    # ═══════════════════════════════════════════════════════════
    # 公共入口
    # ═══════════════════════════════════════════════════════════

    def run(self,
            folder_path,
            year,
            author,
            asr_model_name,
            available_models,
            locate_model_key=None,
            locate_sys_prompt="",
            locate_user_prompt_template="",
            anchor_sys_prompt="",
            anchor_user_prompt_template="",
            check_model_key=None,
            check_sys_prompt="",
            check_user_prompt_template="",
            anchor_model_key=None,
            last_n_chars=1000,
            enable_mark_quotes=True,
            enable_miss_check=True,
            min_chars_per_item=80,
            debug=False):

        state = self._setup_state(
            folder_path, asr_model_name, available_models,
            anchor_model_key, locate_model_key, check_model_key, debug,
        )

        # [0/8] ASR 转录
        transcript_text = step_asr(self, state)

        # [1/8] 模糊原文
        fuzzy_ref, duration = step_fuzzy_text(self, state)

        # [2/8] 脚注重排
        fuzzy_ref = step_rearrange(self, state, fuzzy_ref)

        # [3/8] + [4/8] 末尾定位 → 精准原文
        precision_text, precise_body = step_locate_and_precision(
            self, state, fuzzy_ref, transcript_text,
            locate_sys_prompt, locate_user_prompt_template, last_n_chars,
        )

        # [5/8] 编号锚点校对
        raw_output, verify_ok, verify_msg = step_anchor_proofread(
            self, state, transcript_text, precision_text,
            anchor_sys_prompt, anchor_user_prompt_template,
            min_chars_per_item,
        )

        # [6/8] 遗漏检查
        no_loss = None
        check_text = None
        if enable_miss_check:
            no_loss, check_text = step_miss_check(
                self, state, transcript_text, raw_output,
                check_sys_prompt, check_user_prompt_template,
            )

        # [7/8] 脚本去编号
        final_text = step_remove_numbers(self, state, raw_output)

        # [8/8] 插入元数据 + 标注 + 汇总
        step_finalize(
            self, state, final_text, precise_body, transcript_text,
            year, author, duration, enable_mark_quotes,
            verify_ok, verify_msg, no_loss, check_text,
        )

    # ═══════════════════════════════════════════════════════════
    # Setup
    # ═══════════════════════════════════════════════════════════

    def _setup_state(self, folder_path, asr_model_name,
                     available_models, anchor_model_key,
                     locate_model_key, check_model_key, debug):
        """初始化所有文件路径、引擎、计时变量，返回状态字典。"""
        s: dict[str, Any] = {
            "debug": debug,
            "total_time": 0.0,
            "step_times": [],
        }

        if not folder_path.exists():
            print(f"{Color.RED}错误：找不到文件夹 {folder_path}")
            exit(1)

        audio_files = list(folder_path.glob("*.mp3"))
        if not audio_files:
            print(f"{Color.RED}错误：在 {folder_path} 下没找到 .mp3 文件")
            exit(1)

        audio_path_obj = next(
            (f for f in audio_files if f.stem == folder_path.name),
            audio_files[0],
        )
        s["audio_path"] = str(audio_path_obj)
        s["base_name"] = audio_path_obj.stem
        s["parent_dir"] = audio_path_obj.parent

        print(f"{Color.GREEN}已自动识别任务：{Color.END}")
        print(f"  - 文件夹: {folder_path}")
        print(f"  - 音频文件: {audio_path_obj.name}")

        p = s["parent_dir"]
        s["files"] = {
            "fuzzy":           p / "第1步_模糊原文.md",
            "transcript":      p / f"{s['base_name']}_逐字稿.md",
            "precision":       p / f"第4步_{s['base_name']}_精准原文.md",
            "precise_body":    p / "第4步_精准原文_正文.md",
            "numbered":        p / f"第5步_{s['base_name']}_编号稿.md",
            "anchor_output":   p / f"第5步_{s['base_name']}_编号校对稿.md",
            "check":           p / f"第6步_{s['base_name']}_遗漏检查.md",
            "final":           p / f"第7步_{s['base_name']}_最终校对稿.md",
        }

        s["available_models"] = available_models
        s["anchor_model_key"] = anchor_model_key
        s["locate_model_key"] = locate_model_key
        s["check_model_key"] = check_model_key

        print("正在初始化 Qwen3ASRFlashFiletrans 引擎...")
        s["asr_engine"] = Qwen3ASRFlashFiletrans(model_name=asr_model_name)

        return s

    # ═══════════════════════════════════════════════════════════
    # Helpers（供步骤模块调用）
    # ═══════════════════════════════════════════════════════════

    def _init_model(self, s, label, key):
        """初始化一个 LLM 引擎（带打印）。"""
        cls, kwargs = s["available_models"][key]
        print(f"正在初始化 {cls.__name__} 引擎 ({key})...")
        return cls(**kwargs)

    def _llm_generate(self, s, engine, sys_prompt, prompt,
                       step_label, think_prefix, think_label):
        """通用 LLM 调用：计时 → 调用 → 保存思维链 → 返回 (result, elapsed)。"""
        print("调用中...")
        t0 = time.time()
        result = "".join(engine.generate_stream(sys_prompt, prompt))
        elapsed = time.time() - t0
        s["total_time"] += elapsed
        s["step_times"].append((step_label, elapsed))
        self._save_thinking(s, engine, think_prefix, think_label)
        label_tail = step_label.split('] ')[-1] if '] ' in step_label else step_label
        print(f"⏱️ {label_tail}耗时: {format_duration(elapsed)}")
        return result, elapsed

    def _call_anchor_llm(self, s, engine, sys_prompt, prompt_template,
                          numbered_text, total_sentences, precision_text,
                          step_label):
        """调用编号校对 LLM，保存结果，返回 raw_output。"""
        self._confirm_step("编号锚点校对", s["debug"])
        prompt = prompt_template.format(
            transcript_text=numbered_text,
            total_sentences=total_sentences,
            precision_text=precision_text,
        )
        print(f"正在调用 {engine.model_name} 进行编号锚点校对...")
        result, _ = self._llm_generate(
            s, engine, sys_prompt, prompt,
            step_label, "第5步_", "编号锚点校对",
        )
        s["files"]["anchor_output"].write_text(result, encoding="utf-8")
        return result

    def _save_thinking(self, s, engine, step_prefix, label):
        """保存 LLM 思维链到文件。"""
        if hasattr(engine, '_last_reasoning_content') and engine._last_reasoning_content:
            path = s["parent_dir"] / f"{step_prefix}思维链_{label}.md"
            path.write_text(engine._last_reasoning_content, encoding="utf-8")
            engine._last_reasoning_content = None
            print(f"{Color.GREEN}🧠 思维链已保存: {path.name}{Color.END}")

    def _build_numbered_draft(self, transcript_text, min_chars_per_item):
        """将逐字稿按标点切分并合并为编号稿。返回 (numbered_text, raw_sentences)。"""
        segments = re.split(r'(?<=[。！？，、；：])', transcript_text)
        segments = [x.strip() for x in segments if x.strip()]
        sentences = []
        buf = ""
        for seg in segments:
            buf += seg
            if len(buf) >= min_chars_per_item:
                sentences.append(buf)
                buf = ""
        if buf:
            sentences.append(buf)
        lines = [f"[{i + 1:04d}] {s}" for i, s in enumerate(sentences)]
        return "\n".join(lines), sentences

    def _verify_numbering(self, raw_output, total_sentences):
        """验证编号完整性。返回 (verify_ok, verify_msg, duplicate_list)。"""
        print(f"\n--- 编号完整性验证 ---")
        numbers = re.findall(r'\[(\d{4})\]', raw_output)
        if not numbers:
            msg = "❌ 输出中未检测到任何编号"
            print(msg)
            print("----------------------------")
            return False, msg, []

        nums = [int(n) for n in numbers]
        missing = [i for i in range(1, total_sentences + 1) if nums.count(i) == 0]
        duplicate = [(i, nums.count(i)) for i in range(1, total_sentences + 1) if nums.count(i) > 1]

        if not missing and not duplicate:
            msg = f"编号序列完整，全部 {total_sentences} 句均已输出"
            print(f"{Color.GREEN}✅ {msg}{Color.END}")
            print("----------------------------")
            return True, msg, []

        parts = []
        if missing:
            parts.append(f"缺失编号 ({len(missing)} 个): {missing[:20]}{'...' if len(missing) > 20 else ''}")
        if duplicate:
            parts.append(f"重复编号: {[f'{n}(×{c})' for n, c in duplicate[:10]]}")
        msg = "; ".join(parts)
        print(f"⚠️ {msg}")
        print("----------------------------")
        return False, msg, duplicate

    def _extract_precise_body(self, precision_text):
        """从精准原文中提取摩诃止观正文（** 包裹的行）。"""
        clean_text = re.sub(r'\[注脚：.*?\]', '', precision_text, flags=re.DOTALL)
        parts = []
        for line in clean_text.splitlines():
            if "**" in line:
                clean = re.sub(r'[\*\\]', '', line).strip()
                if clean:
                    parts.append(clean)
        return "\n\n".join(parts)

    def _load_or_extract_precise_body(self, s, precision_text):
        """获取精准原文正文（优先缓存）。"""
        fn = s["files"]["precise_body"]
        if fn.exists():
            result = fn.read_text("utf-8").strip()
            print(f"{Color.GREEN}✅ 检测到已存在精准原文正文，直接复用: {fn.name}{Color.END}")
            return result
        return self._extract_precise_body(precision_text)

    @staticmethod
    def _should_regenerate_concentration(a_ratio):
        """浓缩率不在 70%-95% 区间时询问用户。返回 True 表示需要重新生成。"""
        if 70.0 <= a_ratio <= 95.0:
            return False
        reason = "过低" if a_ratio < 70.0 else "过高"
        choice = input(
            f"{Color.ORANGE}⚠️ 浓缩率 {a_ratio:.1f}% {reason}，是否重新生成编号校对稿？(y/n): {Color.END}"
        ).strip().lower()
        return choice == "y"

    @staticmethod
    def _merge_sentences(parts):
        """合并短句：buf 累加到句号结尾时 flush。"""
        merged = []
        buf = ""
        for c in parts:
            buf = buf + c if buf else c
            if buf and buf[-1] in '。！？':
                merged.append(buf)
                buf = ""
        if buf:
            merged.append(buf)
        return merged

    @staticmethod
    def _confirm_step(label, debug):
        """调试模式：暂停等待用户确认。"""
        if not debug:
            return
        choice = input(f"🔍 调试模式：即将执行【{label}】，输入 y 继续(n 退出): ").strip().lower()
        if choice != "y":
            print(f"{Color.RED}❌ 用户终止。{Color.END}")
            exit(0)
        print(f"{Color.GREEN}✅ 继续执行【{label}】...{Color.END}")
