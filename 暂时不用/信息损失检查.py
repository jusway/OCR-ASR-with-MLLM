import os
import sys
from pathlib import Path
from core.text_engine import GeminiText, NvidiaText
from core.utils import Color


class InfoLossVerifier:
    """信息损失检查器：对比逐字稿与校对稿，确保没有关键信息丢失"""

    def __init__(self, text_engine):
        self.text_engine = text_engine

    def verify(self, transcript_path, written_path,
               verifier_sys_prompt, verifier_user_prompt_template,
               force=False):
        transcript_path = Path(transcript_path)
        written_path = Path(written_path)

        # 自动生成检查报告的文件名
        # 规则：基于校对稿文件名，将 _校对稿 替换为 _丢失信息检查
        report_filename = written_path.parent / f"{written_path.stem.replace('_校对稿', '')}_丢失信息检查.md"

        print(f"\n{Color.CYAN}==========================================")
        print(f"🔍 正在检查: {written_path.name}")
        print(f"=========================================={Color.END}")

        if report_filename.exists() and not force:
            print(f"{Color.GREEN}✅ 检查报告已存在，跳过本次生成: {report_filename.name}{Color.END}")
            with open(report_filename, "r", encoding="utf-8") as f:
                return f.read()

        if not transcript_path.exists():
            print(f"{Color.RED}❌ 错误：找不到逐字稿文件 {transcript_path}{Color.END}")
            return None
        if not written_path.exists():
            print(f"{Color.RED}❌ 错误：找不到校对稿文件 {written_path}{Color.END}")
            return None

        print(f"{Color.DARK_PURPLE}正在读取文件内容...{Color.END}")
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript_text = f.read()
        with open(written_path, "r", encoding="utf-8") as f:
            written_text = f.read()

        verifier_user_prompt = verifier_user_prompt_template.format(
            transcript_text=transcript_text,
            written_text=written_text
        )

        print(f"{Color.DARK_PURPLE}正在调用 AI 引擎进行深度对比（这可能需要一分钟左右）...{Color.END}")
        try:
            report = "".join(self.text_engine.generate_stream(verifier_sys_prompt, verifier_user_prompt))

            with open(report_filename, "w", encoding="utf-8") as f:
                f.write(report)

            print(f"{Color.GREEN}✅ 检查完成！报告已保存至: {report_filename}{Color.END}")

            if "【无遗漏信息】" in report:
                print(f"{Color.GREEN}🎉 结论：AI 认为该文稿完整保留了原始信息。{Color.END}")
            else:
                print(f"{Color.ORANGE}⚠️ 发现可能的信息丢失，请重点关注：{Color.END}")
                # 打印报告的前面几行
                summary_lines = report.split("\n")[:15]
                print("\n".join(summary_lines))
                if len(report.split("\n")) > 15:
                    print("...")

            return report
        except Exception as e:
            print(f"{Color.RED}❌ 检查过程中发生错误: {e}{Color.END}")
            return None

    def scan_and_verify(self, root_dir,
                        verifier_sys_prompt, verifier_user_prompt_template,
                        force=False):
        """扫描目录下所有的 逐字稿/校对稿 对并执行检查"""
        root_path = Path(root_dir)
        written_files = list(root_path.rglob("*_校对稿.md"))

        if not written_files:
            print(f"{Color.ORANGE}在目录 {root_dir} 下未找到任何校对稿文件。{Color.END}")
            return

        print(f"{Color.CYAN}发现 {len(written_files)} 个校对稿，准备开始批量检查...{Color.END}")

        for written_path in written_files:
            # 根据校对稿找对应的逐字稿
            transcript_path = written_path.parent / written_path.name.replace("_校对稿.md", "_逐字稿.md")
            self.verify(transcript_path, written_path,
                        verifier_sys_prompt, verifier_user_prompt_template,
                        force=force)


if __name__ == "__main__":
    # ============================================================
    # ================ 用户配置区（按需修改即可） ================
    # ============================================================

    # 1) 模式选择：'single' 为单个文件夹模式, 'batch' 为批量扫描模式
    mode = 'batch'

    # 2) [单文件模式] 指定文件夹路径（会自动寻找该目录下的逐字稿和校对稿）
    target_folder = "摩诃止观-久仁法师/摩诃止观001"

    # 3) [批量模式] 指定根目录（会递归搜索所有子文件夹）
    batch_root_dir = "摩诃止观-久仁法师"

    # 4) 是否强制重新检查（即使已经有报告了也重新生成）
    force_recheck = False

    # 5) 文本模型：想换模型只改 SELECTED_MODEL 这一行的 key
    #    新增模型：照样追加一行 "别名": (引擎类, {"model_name": "...", ...其它初始化参数}),
    AVAILABLE_MODELS = {
        "kimi":     (NvidiaText, {"model_name": "moonshotai/kimi-k2-instruct-0905", "temperature": 1.0, "top_p": 1.0}),
        "minimax":  (NvidiaText, {"model_name": "minimaxai/minimax-m2.7",           "temperature": 1.0, "top_p": 1.0}),
        "deepseek": (NvidiaText, {"model_name": "deepseek-ai/deepseek-v4-pro",      "temperature": 1.0, "top_p": 1.0}),
        "gemini":   (GeminiText, {"model_name": "gemini-3.1-pro-preview",           "temperature": 1.0}),
    }
    SELECTED_MODEL = "gemini"

    # 6) 信息丢失检查提示词（与 一键生成校对稿.py 保持一致）
    verifier_system_prompt = (
        "你是一位严谨校对员。你的任务是对比【原始逐字稿】和【校对稿】，检查校对稿是否丢失了逐字稿中的关键信息。"
        "如果没有遗漏，直接输出【无遗漏信息】。"
        "如果有遗漏，请解释。"
    )
    verifier_user_prompt_template = (
        "【原始逐字稿】\n{transcript_text}\n\n"
        "【校对稿】\n{written_text}\n"
    )

    # ============================================================
    # ================ 以下为流程代码，无需改动 ==================
    # ============================================================

    engine_cls, engine_kwargs = AVAILABLE_MODELS[SELECTED_MODEL]
    print(f"{Color.DARK_PURPLE}正在初始化 {engine_cls.__name__} 引擎 ({SELECTED_MODEL} -> {engine_kwargs['model_name']})...{Color.END}")
    text_engine = engine_cls(**engine_kwargs)
    verifier = InfoLossVerifier(text_engine)

    if mode == 'single':
        folder = Path(target_folder)
        # 寻找该文件夹下的校对稿
        written_files = list(folder.glob("*_校对稿.md"))
        if not written_files:
            print(f"{Color.RED}❌ 错误：在文件夹 {target_folder} 中没找到校对稿。{Color.END}")
        else:
            for wp in written_files:
                tp = wp.parent / wp.name.replace("_校对稿.md", "_逐字稿.md")
                verifier.verify(tp, wp,
                                verifier_system_prompt, verifier_user_prompt_template,
                                force=force_recheck)

    elif mode == 'batch':
        verifier.scan_and_verify(batch_root_dir,
                                 verifier_system_prompt, verifier_user_prompt_template,
                                 force=force_recheck)

    else:
        print(f"{Color.RED}❌ 未知的模式: {mode}{Color.END}")
