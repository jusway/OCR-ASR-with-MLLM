"""
精准末尾定位校对流水线。
要点：保证模糊原文开头正确，只需定位末尾，Python截断。

流程：
1) ASR稿件后1000字 + 模糊原文 → LLM → 末尾
2) Python截断模糊原文（开头 → 末尾）→ 精准原文
3) ASR稿件 + 精准原文 → LLM → 最终校对稿
4) 精准原文模式匹配 → 摩诃止观正文 → 插入元数据
"""
import re
import subprocess
import time
from pathlib import Path

from core.utils import Color
from core.asr_engine import Qwen3ASRFlashFiletrans
from 常用脚本.脚注重排 import rearrange_footnotes


class PreciseLocatePipeline:
    """精准末尾定位流水线。一步调用，内部完成文件发现、引擎初始化、四步执行。"""

    def run(self,
            folder_path,
            year,
            author,
            asr_model_name,
            available_models,
            locate_model_key,
            final_model_key,
            draft_model_key=None,
            draft_sys_prompt="",
            draft_user_prompt_template="",
            example_text="",
            locate_sys_prompt="",
            locate_user_prompt_template="",
            final_sys_prompt="",
            final_user_prompt_template="",
            check_model_key=None,
            check_sys_prompt="",
            check_user_prompt_template="",
            last_n_chars=1000,
            debug=False):

        # ── 初始化引擎 ─────────────────────
        def _init(label, key):
            cls, kwargs = available_models[key]
            print(f"正在初始化 {cls.__name__} 引擎 ({key})...")
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
        base_name = audio_path_obj.stem
        parent_dir = audio_path_obj.parent

        print(f"已自动识别任务：")
        print(f"  - 文件夹: {folder_path}")
        print(f"  - 音频文件: {audio_path_obj.name}")

        fuzzy_filename = parent_dir / "第1步_模糊原文.md"
        transcript_filename = parent_dir / f"{base_name}_逐字稿.md"
        draft_filename = parent_dir / f"第0.5步_{base_name}_初次校对稿.md"
        precision_filename = parent_dir / f"第1步_{base_name}_精准原文.md"
        final_filename = parent_dir / f"第2步_{base_name}_最终校对稿.md"
        # 摩诃止观正文不再单独建文件，参见 精准原文_正文.md

        # ── 前置：模糊原文 ──────────────────
        if not fuzzy_filename.exists():
            fuzzy_filename.write_text("", encoding="utf-8")
            print(f"{Color.GREEN}✅ 已创建 {fuzzy_filename}{Color.END}")
        else:
            print(f"检测到已存在 {fuzzy_filename.name}")
        duration = self._get_audio_duration(audio_path)
        print(f"{Color.ORANGE}⏱️ 音频时长: {duration}，供参考选取原文范围{Color.END}")
        print(f"请在 {fuzzy_filename.name} 中整理模糊原文（开头已有上节课结尾，只需粘贴即可），保存后回到此处按回车继续...")
        input()
        fuzzy_reference_text = fuzzy_filename.read_text("utf-8").strip()
        if not fuzzy_reference_text:
            print(f"{Color.RED}错误：模糊原文为空，请填入内容后重新运行。")
            exit(1)

        # ── 第1步预处理：脚注重排 ──────
        rearranged_fn = parent_dir / "第1步_模糊原文_已重排.md"
        if rearranged_fn.exists():
            fuzzy_reference_text = rearranged_fn.read_text("utf-8").strip()
            print(f"{Color.GREEN}✅ 检测到已存在重排后模糊原文，直接复用: {rearranged_fn.name}{Color.END}")
        else:
            _t0 = time.time()
            fuzzy_reference_text = rearrange_footnotes(fuzzy_reference_text)
            rearranged_fn.write_text(fuzzy_reference_text, encoding="utf-8")
            print(f"{Color.GREEN}✅ 脚注重排完成 → {rearranged_fn.name}{Color.END}")

        # ── 初始化引擎 ────────────
        print("正在初始化 Qwen3ASRFlashFiletrans 引擎...")
        asr_engine = Qwen3ASRFlashFiletrans(model_name=asr_model_name)

        _total_time = 0.0

        def _save_thinking(step_prefix, engine, label):
            if hasattr(engine, '_last_reasoning_content') and engine._last_reasoning_content:
                thinking_path = parent_dir / f"{step_prefix}思维链_{label}.md"
                thinking_path.write_text(engine._last_reasoning_content, encoding="utf-8")
                engine._last_reasoning_content = None
                print(f"{Color.GREEN}🧠 思维链已保存: {thinking_path.name}{Color.END}")

        # ── 步骤0：ASR 转录（带缓存）─────
        print(f"\n[0/4] 开始处理整段音频转录...")
        if transcript_filename.exists():
            print(f"{Color.GREEN}✅ 检测到已存在逐字稿，直接复用: {transcript_filename.name}{Color.END}")
            transcript_text = transcript_filename.read_text("utf-8")
        else:
            print("正在调用 ASR 引擎生成逐字稿...")
            _t0 = time.time()
            transcript_text = asr_engine.recognize(audio_path)
            _elapsed = time.time() - _t0
            _total_time += _elapsed
            print(f"⏱️ ASR 转录耗时: {self._format_time(_elapsed)}")
            transcript_filename.write_text(transcript_text, encoding="utf-8")
            print(f"{Color.GREEN}逐字稿已保存至: {transcript_filename}{Color.END}")

        # ── 步骤0.5：初次校对稿 ──────────
        draft_text = transcript_text  # 默认不校对
        draft_engine = _init("初次校对", draft_model_key) if debug and draft_model_key else None
        if draft_engine and draft_sys_prompt and draft_user_prompt_template:
            if draft_filename.exists():
                print(f"\n[0.5/4] {Color.GREEN}✅ 检测到已存在初次校对稿，直接复用: {draft_filename.name}{Color.END}")
                draft_text = draft_filename.read_text("utf-8")
                draft_ratio = len(draft_text) / len(transcript_text) * 100
                print(f"{Color.ORANGE}📊 字数统计：初次校对稿 {len(draft_text)} / 逐字稿 {len(transcript_text)} = {draft_ratio:.1f}%{Color.END}")
            else:
                print(f"\n[0.5/4] 正在生成初次校对稿（口语转书面）...")
                self._confirm_step("初次校对", debug)
                draft_prompt = draft_user_prompt_template.format(
                    transcript_text=transcript_text,
                    example_text=example_text,
                )
                print("调用中...")
                _t0 = time.time()
                draft_text = draft_engine.generate(draft_sys_prompt, draft_prompt)
                _elapsed = time.time() - _t0
                _total_time += _elapsed
                _save_thinking("第0.5步_", draft_engine, "初次校对")
                print(f"⏱️ 初次校对耗时: {self._format_time(_elapsed)}")
                if draft_text.startswith("【校对稿】"):
                    draft_text = draft_text[len("【校对稿】"):].strip()
                    print(f"{Color.GREEN}✅ 后处理：已去掉【校对稿】前缀{Color.END}")
                draft_filename.write_text(draft_text, encoding="utf-8")
                print(f"{Color.GREEN}初次校对稿已保存至: {draft_filename.name}{Color.END}")
                draft_ratio = len(draft_text) / len(transcript_text) * 100
                print(f"{Color.ORANGE}📊 字数统计：初次校对稿 {len(draft_text)} / 逐字稿 {len(transcript_text)} = {draft_ratio:.1f}%{Color.END}")

        # ── 步骤1：末尾定位 ────────────
        print(f"\n[1/4] 正在定位讲稿末尾对应的原文...")
        locate_engine = _init("末尾定位", locate_model_key)
        no_body_extract = False
        if precision_filename.exists():
            print(f"{Color.GREEN}✅ 检测到已存在精准原文，直接复用: {precision_filename.name}{Color.END}")
            precision_text = precision_filename.read_text("utf-8")
        else:
            self._confirm_step("末尾定位", debug)
            tail_text = draft_text[-last_n_chars:]
            locate_prompt = locate_user_prompt_template.format(
                tail_text=tail_text,
                fuzzy_reference_text=fuzzy_reference_text,
                last_n_chars=last_n_chars,
            )
            print("调用中...")
            _t0 = time.time()
            last_sentence = locate_engine.generate(locate_sys_prompt, locate_prompt).strip()
            _elapsed = time.time() - _t0
            _total_time += _elapsed
            _save_thinking("第1步_", locate_engine, "末尾定位")
            print(f"⏱️ 末尾定位耗时: {self._format_time(_elapsed)}")
            # 模型返回末尾结果已保存至文件，不在控制台重复打印
            locate_out_filename = parent_dir / f"第1步_{base_name}_末尾定位.md"
            locate_out_filename.write_text(last_sentence, encoding="utf-8")
            print(f"{Color.GREEN}✅ 末尾已保存至: {locate_out_filename.name}{Color.END}")

            # 模型无法匹配的特殊标记 → 精准原文和正文置空，流程继续
            if last_sentence == "【无匹配片段】":
                print(f"{Color.ORANGE}⚠️ 模型返回「无匹配片段」，精准原文和正文将为空。{Color.END}")
                precision_text = "暂无参考资料"
                precise_body = ""
                no_body_extract = True
            else:
                no_body_extract = False
                # 在模糊原文中找到这句的边界
                idx = fuzzy_reference_text.find(last_sentence)
                if idx == -1:
                    print(f"{Color.RED}❌ 未在模糊原文中找到末尾，模型返回的结果不正确。请检查末尾定位结果后重试。{Color.END}")
                    print(f"{Color.RED}末尾定位文件: {locate_out_filename.name}{Color.END}")
                    exit(1)
                end_pos = idx + len(last_sentence)
                precision_text = fuzzy_reference_text[:end_pos]

            # 字数对比省略
            precision_filename.write_text(precision_text, encoding="utf-8")
            print(f"{Color.GREEN}✅ 精准原文已保存至: {precision_filename.name}{Color.END}")

        # ── 提取精准原文中的摩诃止观正文 ──
        precise_body_fn = parent_dir / "第1步_精准原文_正文.md"
        if no_body_extract:
            # 已在上面置空，跳过提取，只写文件
            precise_body_fn.write_text(precise_body, encoding="utf-8")
        elif precise_body_fn.exists():
            precise_body = precise_body_fn.read_text("utf-8").strip()
            print(f"{Color.GREEN}✅ 检测到已存在精准原文正文，直接复用: {precise_body_fn.name}{Color.END}")
        else:
            # 先剔除 [注脚：...] 块（含跨行），再扫 *** 提取正文
            clean_text = re.sub(r'\[注脚：.*?\]', '', precision_text, flags=re.DOTALL)
            precise_body_parts = []
            for line in clean_text.splitlines():
                if "**" in line:
                    clean = re.sub(r'[\*\\]', '', line).strip()
                    if clean:
                        precise_body_parts.append(clean)
            precise_body = "\n\n".join(precise_body_parts)
            precise_body_fn.write_text(precise_body, encoding="utf-8")
            print(f"{Color.GREEN}✅ 精准原文正文已提取至: {precise_body_fn.name}（{len(precise_body)} 字）{Color.END}")

        # ── 步骤2：最终校对稿 ──────────
        print(f"\n[2/4] 正在生成最终校对稿...")
        final_engine = _init("最终校对", final_model_key)
        if final_filename.exists():
            print(f"{Color.GREEN}✅ 检测到已存在最终校对稿，直接复用: {final_filename.name}{Color.END}")
            final_text = final_filename.read_text("utf-8")
            _ratio_cache = len(final_text) / len(transcript_text) * 100
            print(f"{Color.ORANGE}📊 字数统计：最终校对稿 {len(final_text)} / 逐字稿 {len(transcript_text)} = {_ratio_cache:.1f}%{Color.END}")
        else:
            self._confirm_step("最终校对（大模型）", debug)
            final_prompt = final_user_prompt_template.format(
                precision_text=precision_text,
                transcript_text=transcript_text,
            )
            print("调用中...")
            _t0 = time.time()
            final_text = final_engine.generate(final_sys_prompt, final_prompt)
            _elapsed = time.time() - _t0
            _total_time += _elapsed
            _save_thinking("第2步_", final_engine, "最终校对")
            print(f"⏱️ 最终校对稿模型调用耗时: {self._format_time(_elapsed)}")
            if final_text.startswith("【校对稿】"):
                final_text = final_text[len("【校对稿】"):].strip()
                print(f"{Color.GREEN}✅ 后处理：已去掉【校对稿】前缀{Color.END}")

            # 浓缩率检查（低于50%重试）
            for retry in range(3):
                _ratio = len(final_text) / len(transcript_text) * 100
                print(f"{Color.ORANGE}📊 字数统计：校对稿 {len(final_text)} / 逐字稿 {len(transcript_text)} = {_ratio:.1f}%{Color.END}")
                if _ratio >= 50.0 or retry == 2:
                    break
                print(f"{Color.ORANGE}⚠️ 浓缩率 {_ratio:.1f}% 低于 50%，第 {retry + 2} 次重试...{Color.END}")
                print("调用中...")
                _t0 = time.time()
                final_text = final_engine.generate(final_sys_prompt, final_prompt)
                _elapsed = time.time() - _t0
                _total_time += _elapsed
                _save_thinking("第2步_", final_engine, "最终校对")
                if final_text.startswith("【校对稿】"):
                    final_text = final_text[len("【校对稿】"):].strip()

            final_filename.write_text(final_text, encoding="utf-8")
            print(f"{Color.GREEN}最终校对稿已保存至: {final_filename.name}{Color.END}")

        # ── 步骤3：遗漏检查 ──────────────
        check_filename = parent_dir / f"第3步_{base_name}_遗漏检查.md"
        print(f"\n[3/4] 正在进行遗漏检查...")
        check_engine = _init("遗漏检查", check_model_key) if check_model_key else None
        if check_engine and check_sys_prompt and check_user_prompt_template:
            if check_filename.exists():
                print(f"\n[3/4] {Color.GREEN}✅ 检测到已存在遗漏检查报告，直接复用: {check_filename.name}{Color.END}")
            else:
                print(f"\n[3/4] 正在进行遗漏检查（ASR逐字稿 vs 最终校对稿）...")
                self._confirm_step("遗漏检查", debug)
                check_prompt = check_user_prompt_template.format(
                    transcript_text=transcript_text,
                    written_text=final_text,
                )
                print("调用中...")
                _t0 = time.time()
                check_report = check_engine.generate(check_sys_prompt, check_prompt)
                _elapsed = time.time() - _t0
                _total_time += _elapsed
                _save_thinking("第3步_", check_engine, "遗漏检查")
                print(f"⏱️ 遗漏检查耗时: {self._format_time(_elapsed)}")
                check_filename.write_text(check_report, encoding="utf-8")
                print(f"{Color.GREEN}✅ 遗漏检查报告已保存至: {check_filename.name}{Color.END}")
                # 根据报告内容标记状态
                if "【无遗漏信息】" in check_report or "完美" in check_report:
                    print(f"{Color.GREEN}  ✓ 检查结果：无关键信息遗漏{Color.END}")
                else:
                    print(f"{Color.ORANGE}  ⚠️ 检查结果：可能有遗漏，请查看报告{Color.END}")

        # ── 步骤4：插入元数据 ───────────
        main_body = precise_body  # 复用已提取的精准原文正文
        print(f"\n[4/4] 准备补充元数据信息...")
        current_text = final_filename.read_text("utf-8")
        if current_text.strip().startswith("> 标题："):
            parts = current_text.split("\n---\n\n", 1)
            current_text = parts[1] if len(parts) > 1 else current_text
            print(f"{Color.ORANGE}♻️ 检测到已有旧元数据，替换为本次元数据...{Color.END}")
        duration = self._get_audio_duration(audio_path)
        main_body_header = f"> 摩诃止观正文：{main_body.replace(chr(10), '')}\n" if main_body else ""
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
            print("📄 以下为插入的摩诃止观正文：")
            print(main_body_header.replace("> 摩诃止观正文：", "").strip())

        print(f"⏱️ 本次全部模型调用总耗时: {self._format_time(_total_time)}")
        print(f"\n===========================================")
        print(f"{Color.GREEN}🎉 全部流程处理完成！{Color.END}")
        print(f"💡 最终文件列表：")
        print(f"  - 逐字稿: {transcript_filename.name}")
        print(f"  - 初次校对稿: {draft_filename.name}")
        print(f"  - 精准原文: {precision_filename.name}")
        print(f"  - 最终校对稿: {final_filename.name}")
        print(f"  - 摩诃止观正文: 第1步_精准原文_正文.md")
        print(f"===========================================\n")

    # ── 工具方法 ─────────────────────────────

    @staticmethod
    def _confirm_step(label, debug):
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
    def _format_time(seconds):
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        if minutes > 0:
            return f"{minutes}分{secs}秒"
        return f"{secs}秒"
