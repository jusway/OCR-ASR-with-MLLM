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
import subprocess
import time
from pathlib import Path

from core.utils import Color
from core.asr_engine import Qwen3ASRFlashFiletrans
from 常用脚本.脚注重排 import rearrange_footnotes


class AnchorProofreadPipeline:
    """编号锚点校对流水线。融合精准末端定位 + 编号锚点防遗漏机制。"""

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
            debug=False):

        # ── 初始化引擎 ─────────────────────
        def _init(label, key):
            cls, kwargs = available_models[key]
            print(f"正在初始化 {cls.__name__} 引擎 ({key})...")
            return cls(**kwargs)

        if not folder_path.exists():
            print(f"{Color.RED}错误：找不到文件夹 {folder_path}")
            exit(1)

        audio_files = list(folder_path.glob("*.mp3"))
        if not audio_files:
            print(f"{Color.RED}错误：在 {folder_path} 下没找到 .mp3 文件")
            exit(1)
        audio_path_obj = next((f for f in audio_files if f.stem == folder_path.name), audio_files[0])
        audio_path = str(audio_path_obj)
        base_name = audio_path_obj.stem
        parent_dir = audio_path_obj.parent

        print(f"{Color.GREEN}已自动识别任务：{Color.END}")
        print(f"  - 文件夹: {folder_path}")
        print(f"  - 音频文件: {audio_path_obj.name}")

        fuzzy_filename = parent_dir / "第1步_模糊原文.md"
        transcript_filename = parent_dir / f"{base_name}_逐字稿.md"
        precision_filename = parent_dir / f"第4步_{base_name}_精准原文.md"
        numbered_draft_filename = parent_dir / f"第5步_{base_name}_编号稿.md"
        anchor_output_filename = parent_dir / f"第5步_{base_name}_编号校对稿.md"
        final_filename = parent_dir / f"第7步_{base_name}_最终校对稿.md"
        check_filename = parent_dir / f"第6步_{base_name}_遗漏检查.md"

        _total_time = 0.0
        _step_times = []

        # ── 初始化 ASR 引擎 ────────────
        print(f"正在初始化 Qwen3ASRFlashFiletrans 引擎...")
        asr_engine = Qwen3ASRFlashFiletrans(model_name=asr_model_name)

        def _save_thinking(step_prefix, engine, label):
            if hasattr(engine, '_last_reasoning_content') and engine._last_reasoning_content:
                thinking_path = parent_dir / f"{step_prefix}思维链_{label}.md"
                thinking_path.write_text(engine._last_reasoning_content, encoding="utf-8")
                engine._last_reasoning_content = None
                print(f"{Color.GREEN}🧠 思维链已保存: {thinking_path.name}{Color.END}")

        # ═══════════════════════════════════
        # 步骤 [0/8]：ASR 转录（带缓存）
        # ═══════════════════════════════════
        print(f"\n[0/8] ASR 转录 → 逐字稿.md...")
        if transcript_filename.exists():
            print(f"{Color.GREEN}✅ 检测到已存在逐字稿，直接复用: {transcript_filename.name}{Color.END}")
            transcript_text = transcript_filename.read_text("utf-8")
        else:
            print(f"正在调用 ASR 引擎生成逐字稿...")
            _t0 = time.time()
            transcript_text = asr_engine.recognize(audio_path)
            _elapsed = time.time() - _t0
            _total_time += _elapsed
            _step_times.append(("[0/8] ASR 转录", _elapsed))
            print(f"⏱️ ASR 转录耗时: {self._format_time(_elapsed)}")
            transcript_filename.write_text(transcript_text, encoding="utf-8")
            print(f"{Color.GREEN}逐字稿已保存至: {transcript_filename}{Color.END}")

        # ═══════════════════════════════════
        # 步骤 [1/8]：模糊原文
        # ═══════════════════════════════════
        print(f"\n[1/8] 粘贴模糊原文到 {fuzzy_filename.name}...")
        if not fuzzy_filename.exists():
            fuzzy_filename.write_text("", encoding="utf-8")
            print(f"{Color.GREEN}✅ 已创建 {fuzzy_filename}{Color.END}")
        else:
            print(f"检测到已存在 {fuzzy_filename.name}")
        duration = self._get_audio_duration(audio_path)
        print(f"⏱️ 音频时长: {duration}，供参考选取原文范围")
        print(f"请在 {fuzzy_filename.name} 中整理模糊原文（开头已有上节课结尾，只需粘贴即可），保存后回到此处按回车继续...")
        input()
        _start_time = time.time()
        fuzzy_reference_text = fuzzy_filename.read_text("utf-8").strip()
        if not fuzzy_reference_text:
            print(f"{Color.RED}错误：模糊原文为空，请填入内容后重新运行。")
            exit(1)

        # ═══════════════════════════════════
        # 步骤 [2/8]：脚注重排
        # ═══════════════════════════════════
        rearranged_fn = parent_dir / "第2步_模糊原文_已重排.md"
        print(f"\n[2/8] 脚注重排 → {rearranged_fn.name}...")
        if rearranged_fn.exists():
            fuzzy_reference_text = rearranged_fn.read_text("utf-8").strip()
            print(f"{Color.GREEN}✅ 检测到已存在重排后模糊原文，直接复用: {rearranged_fn.name}{Color.END}")
        else:
            _t0 = time.time()
            fuzzy_reference_text = rearrange_footnotes(fuzzy_reference_text)
            rearranged_fn.write_text(fuzzy_reference_text, encoding="utf-8")
            print(f"{Color.GREEN}✅ 脚注重排完成 → {rearranged_fn.name}{Color.END}")

        # ═══════════════════════════════════
        # 步骤 [3/8]：末尾定位 → 精准原文
        # ═══════════════════════════════════
        print(f"\n[3/8] 末尾定位 → 末尾定位.md...")
        locate_engine = _init("末尾定位", locate_model_key)
        no_body_extract = False
        if precision_filename.exists():
            print(f"{Color.GREEN}✅ 检测到已存在精准原文，直接复用: {precision_filename.name}{Color.END}")
            precision_text = precision_filename.read_text("utf-8")
        else:
            self._confirm_step("末尾定位", debug)
            tail_text = transcript_text[-last_n_chars:]
            locate_prompt = locate_user_prompt_template.format(
                tail_text=tail_text,
                fuzzy_reference_text=fuzzy_reference_text,
                last_n_chars=last_n_chars,
            )
            print(f"调用中...")
            _t0 = time.time()
            last_sentence = locate_engine.generate(locate_sys_prompt, locate_prompt).strip()
            _elapsed = time.time() - _t0
            _total_time += _elapsed
            _step_times.append(("[3/8] 末尾定位", _elapsed))
            _save_thinking("第3步_", locate_engine, "末尾定位")
            print(f"⏱️ 末尾定位耗时: {self._format_time(_elapsed)}")
            locate_out_filename = parent_dir / f"第3步_{base_name}_末尾定位.md"
            locate_out_filename.write_text(last_sentence, encoding="utf-8")
            print(f"{Color.GREEN}✅ 末尾已保存至: {locate_out_filename.name}{Color.END}")

            # ── 步骤 [4/8]：精准原文 ──
            print(f"\n[4/8] Python 截断 → 精准原文.md...")
            if last_sentence == "【无匹配片段】":
                print(f"⚠️ 模型返回「无匹配片段」，精准原文和正文将为空。")
                precision_text = "暂无参考资料"
                precise_body = ""
                no_body_extract = True
            else:
                no_body_extract = False
                idx = fuzzy_reference_text.find(last_sentence)
                if idx == -1:
                    print(f"{Color.RED}❌ 未在模糊原文中找到末尾，模型返回的结果不正确。请检查末尾定位结果后重试。{Color.END}")
                    exit(1)
                end_pos = idx + len(last_sentence)
                precision_text = fuzzy_reference_text[:end_pos]

            precision_filename.write_text(precision_text, encoding="utf-8")
            print(f"{Color.GREEN}✅ 精准原文已保存至: {precision_filename.name}{Color.END}")

        # ── 提取精准原文中的摩诃止观正文 ──
        precise_body_fn = parent_dir / "第4步_精准原文_正文.md"
        if no_body_extract:
            precise_body_fn.write_text(precise_body, encoding="utf-8")
        elif precise_body_fn.exists():
            precise_body = precise_body_fn.read_text("utf-8").strip()
            print(f"{Color.GREEN}✅ 检测到已存在精准原文正文，直接复用: {precise_body_fn.name}{Color.END}")
        else:
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

        # ═══════════════════════════════════
        # 步骤 [5/8]：编号锚点校对（核心变更）
        # ═══════════════════════════════════
        print(f"\n[5/8] 编号锚点校对...")

        # -- 5a. 生成编号稿（输入缓存：编号稿存在则复用） --
        if numbered_draft_filename.exists():
            numbered_text = numbered_draft_filename.read_text("utf-8")
            n_ratio = len(numbered_text) / len(transcript_text) * 100
            print(f"{Color.GREEN}✅ 检测到已存在编号稿，直接复用: {numbered_draft_filename.name}{Color.END}")
            print(f"📊 字数统计：编号稿 {len(numbered_text)} / 逐字稿 {len(transcript_text)} = {n_ratio:.1f}%")
            raw_sentences = re.split(r'\n(?=\[\d{4}\])', numbered_text)
            raw_sentences = [s.strip() for s in raw_sentences if s.strip()]
        else:
            raw_sentences = re.split(r'(?<=[。！？])', transcript_text)
            raw_sentences = [s.strip() for s in raw_sentences if s.strip()]
            numbered_lines = []
            for i, s in enumerate(raw_sentences, start=1):
                numbered_lines.append(f"[{i:04d}] {s}")
            numbered_text = "\n".join(numbered_lines)
            numbered_draft_filename.write_text(numbered_text, encoding="utf-8")
            n_ratio = len(numbered_text) / len(transcript_text) * 100
            print(f"{Color.GREEN}✅ 编号稿已生成: {numbered_draft_filename.name}（共 {len(raw_sentences)} 句，字数 {len(numbered_text)} / 逐字稿 {len(transcript_text)} = {n_ratio:.1f}%）{Color.END}")

        total_sentences = len(raw_sentences)

        # -- 5b. LLM 调用（输出缓存：编号校对稿存在则复用） --
        anchor_engine = _init("编号锚点校对", anchor_model_key)
        if anchor_output_filename.exists():
            raw_output = anchor_output_filename.read_text("utf-8")
            a_ratio = len(raw_output) / len(numbered_text) * 100
            print(f"{Color.GREEN}✅ 检测到已存在编号校对稿，直接复用: {anchor_output_filename.name}{Color.END}")
            print(f"📊 浓缩率：编号校对稿 {len(raw_output)} / 编号稿 {len(numbered_text)} = {a_ratio:.1f}%")
        else:
            self._confirm_step("编号锚点校对", debug)
            anchor_prompt = anchor_user_prompt_template.format(
                transcript_text=numbered_text,
                total_sentences=total_sentences,
                precision_text=precision_text,
            )
            print(f"正在调用 {anchor_engine.model_name} 进行编号锚点校对...")
            _t0 = time.time()
            raw_output = anchor_engine.generate(anchor_sys_prompt, anchor_prompt)
            _elapsed = time.time() - _t0
            _total_time += _elapsed
            _step_times.append(("[5/8] 编号锚点校对", _elapsed))
            _save_thinking("第5步_", anchor_engine, "编号锚点校对")
            print(f"⏱️ 编号锚点校对耗时: {self._format_time(_elapsed)}")
            anchor_output_filename.write_text(raw_output, encoding="utf-8")
            print(f"{Color.GREEN}✅ 编号校对稿已保存至: {anchor_output_filename.name}{Color.END}")
            a_ratio = len(raw_output) / len(numbered_text) * 100
            print(f"📊 浓缩率：编号校对稿 {len(raw_output)} / 编号稿 {len(numbered_text)} = {a_ratio:.1f}%")

        # -- 5c. 验证编号完整性 --
        print(f"\n--- 编号完整性验证 ---")
        output_numbers = re.findall(r'\[(\d{4})\]', raw_output)
        if not output_numbers:
            print(f"❌ 输出中未检测到任何编号，LLM 可能未按要求输出。")
        else:
            output_nums_int = [int(n) for n in output_numbers]
            missing = [i for i in range(1, total_sentences + 1) if output_nums_int.count(i) == 0]
            duplicate = [(i, output_nums_int.count(i)) for i in range(1, total_sentences + 1) if output_nums_int.count(i) > 1]
            if not missing and not duplicate:
                print(f"{Color.GREEN}✅ 编号序列完整！全部 {total_sentences} 句均已输出（1:1 对应）{Color.END}")
            else:
                if missing:
                    print(f"⚠️ 缺失编号 ({len(missing)} 个): {missing[:20]}{'...' if len(missing)>20 else ''}")
                if duplicate:
                    print(f"⚠️ 重复编号: {[f'{n}(×{c})' for n, c in duplicate[:10]]}")
            print(f"----------------------------")

        # ═══════════════════════════════════
        # 步骤 [6/8]：遗漏检查
        # ═══════════════════════════════════
        print(f"\n[6/8] 遗漏检查...")
        check_engine = _init("遗漏检查", check_model_key) if check_model_key else None
        if check_engine and check_sys_prompt and check_user_prompt_template:
            if check_filename.exists():
                _no_loss = None  # unknown (cached, can't determine)
                print(f"{Color.GREEN}✅ 检测到已存在遗漏检查报告，直接复用: {check_filename.name}{Color.END}")
            else:
                self._confirm_step("遗漏检查", debug)
                check_prompt = check_user_prompt_template.format(
                    transcript_text=numbered_text,
                    written_text=raw_output,
                )
                print(f"调用中...")
                _t0 = time.time()
                check_report = check_engine.generate(check_sys_prompt, check_prompt)
                _elapsed = time.time() - _t0
                _total_time += _elapsed
                _step_times.append(("[6/8] 遗漏检查", _elapsed))
                _save_thinking("第6步_", check_engine, "遗漏检查")
                print(f"⏱️ 遗漏检查耗时: {self._format_time(_elapsed)}")
                check_filename.write_text(check_report, encoding="utf-8")
                print(f"{Color.GREEN}✅ 遗漏检查报告已保存至: {check_filename.name}{Color.END}")
                if "【无遗漏信息】" in check_report or "完美" in check_report:
                    _no_loss = True
                    print(f"{Color.GREEN}  ✓ 检查结果：无关键信息遗漏{Color.END}")
                else:
                    _no_loss = False
                    print(f"  ⚠️ 检查结果：可能有遗漏，请查看报告")

        # ═══════════════════════════════════
        # 步骤 [7/8]：脚本去编号
        # ═══════════════════════════════════
        print(f"\n[7/8] 脚本去除编号 → 最终校对稿.md...")
        if final_filename.exists():
            print(f"{Color.GREEN}✅ 检测到已存在最终校对稿，直接复用: {final_filename.name}{Color.END}")
            final_text = final_filename.read_text("utf-8")
        else:
            cleaned_parts = []
            for m in re.finditer(r'\[(\d{4})\]\s*', raw_output):
                start = m.end()
                next_match = re.search(r'\[(\d{4})\]\s*', raw_output[start:])
                if next_match:
                    end = start + next_match.start()
                else:
                    end = len(raw_output)
                content = raw_output[start:end].strip()
                if content:
                    cleaned_parts.append(content)
            # 合并未以。！？结尾的不完整段落
            merged = []
            buf = ""
            for c in cleaned_parts:
                if buf:
                    buf += c
                else:
                    buf = c
                if buf and buf[-1] in '。！？':
                    merged.append(buf)
                    buf = ""
            if buf:
                merged.append(buf)
            final_text = "\n\n".join(merged) if merged else raw_output
            final_filename.write_text(final_text, encoding="utf-8")
            print(f"{Color.GREEN}✅ 最终校对稿已保存至: {final_filename.name}{Color.END}")

        # 精准原文正文加粗标记（可选开关）
        if enable_mark_quotes:
            marked_text = self._mark_quotes(final_text, precise_body)
            if marked_text != final_text:
                count = marked_text.count('**') // 2
                print(f"{Color.GREEN}✅ 已在最终校对稿中标注 {count} 处摩诃止观原文引用{Color.END}")
                final_filename.write_text(marked_text, encoding="utf-8")
                final_text = marked_text

        # ═══════════════════════════════════
        print(f"\n[8/8] 插入元数据...")
        current_text = final_filename.read_text("utf-8")
        if current_text.strip().startswith("> 标题："):
            parts = current_text.split("\n---\n\n", 1)
            current_text = parts[1] if len(parts) > 1 else current_text
            print(f"♻️ 检测到已有旧元数据，替换为本次元数据...")
        main_body = precise_body
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

        # ═══════════════════════════════════
        # 汇总
        # ═══════════════════════════════════
        print(f"\n{Color.GREEN}===========================================")
        print(f"🎉 全部流程处理完成！{Color.END}")
        # 字数统计
        clean = final_text
        if "---" in clean:
            clean = clean.split("---", 1)[1]
        clean = clean.replace("**", "").replace("\n", "")
        clean_len = len(clean)
        transcript_len = len(transcript_text)
        final_ratio = clean_len / transcript_len * 100
        print(f"{Color.ORANGE}📊 字数统计：最终校对稿 {clean_len} / 逐字稿 {transcript_len} = {final_ratio:.1f}%{Color.END}")
        # 分步耗时
        print(f"{Color.ORANGE}───────────────────────────────────{Color.END}")
        for step_label, t in _step_times:
            print(f"{Color.ORANGE}⏱️ {step_label}: {self._format_time(t)}{Color.END}")
        print(f"{Color.ORANGE}⏱️ 全部模型调用总耗时: {self._format_time(_total_time)}{Color.END}")
        _elapsed_real = time.time() - _start_time
        print(f"{Color.ORANGE}⏱️ 实际运行总耗时: {self._format_time(_elapsed_real)}{Color.END}")
        print(f"{Color.ORANGE}───────────────────────────────────{Color.END}")
        # 遗漏检查结果
        try:
            _no_loss
        except NameError:
            _no_loss = None
        if _no_loss is True:
            print(f"{Color.GREEN}  ✓ 遗漏检查：无关键信息遗漏{Color.END}")
        elif _no_loss is False:
            print(f"{Color.ORANGE}  ⚠️ 遗漏检查：可能有遗漏，请查看遗漏检查报告{Color.END}")
        print(f"===========================================\n")

    # ── 工具方法 ─────────────────────────────

    @staticmethod
    def _mark_quotes(final_text, precise_body):
        """用 ** 标注 final_text 中与 precise_body 匹配的原文片段。"""
        if not precise_body:
            return final_text

        # 细粒度切分 precise_body（逗号也切，提高短片段匹配率）
        segments = re.split(r'(?<=[。！？，、；])', precise_body)
        segments = [s.strip() for s in segments if s.strip()]
        merged_segments = []
        buf = ""
        for c in segments:
            if buf:
                buf += c
            else:
                buf = c
            if buf and buf[-1] in '。！？':
                merged_segments.append(buf)
                buf = ""
        if buf:
            merged_segments.append(buf)

        # 收集匹配位置
        markers = []
        for seg in merged_segments:
            if len(seg) < 5:
                continue
            idx = final_text.find(seg)
            if idx != -1:
                markers.append((idx, idx + len(seg)))
                continue
            # 最长公共子串扫描
            min_len = max(int(len(seg) * 0.5), 5)
            best = None
            for w in range(len(seg), min_len - 1, -1):
                for i in range(len(seg) - w + 1):
                    sub = seg[i:i + w]
                    idx = final_text.find(sub)
                    if idx != -1:
                        best = (idx, idx + w)
                        break
                if best:
                    break
            if best:
                markers.append(best)

        if not markers:
            return final_text

        # 合并重叠标记
        markers.sort()
        merged = []
        for s, e in markers:
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))

        # 从后往前插入 **
        chars = list(final_text)
        for s, e in reversed(merged):
            chars.insert(e, '**')
            chars.insert(s, '**')
        return ''.join(chars)

    @staticmethod
    def _confirm_step(label, debug):
        if not debug:
            return
        choice = input(f"🔍 调试模式：即将执行【{label}】，输入 y 继续(n 退出): ").strip().lower()
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
