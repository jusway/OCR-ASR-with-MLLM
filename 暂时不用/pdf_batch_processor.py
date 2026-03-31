import time
from pathlib import Path
from typing import Optional
import concurrent.futures
import threading
import requests

from core.ocr_engine import GeminiOCR
from core.pdf_loader import PDFLoader


class PDFBatchProcessor:
    def __init__(self, pdf_path: str, ocr_client: GeminiOCR, system_prompt: str, prompt: str):
        self.pdf_path = Path(pdf_path)
        self.ocr_client = ocr_client
        self.system_prompt = system_prompt
        self.prompt = prompt
        self.loader = PDFLoader(str(self.pdf_path))
        self._pdf_lock = threading.Lock()

    def _process_page(self, page_num: int, output_dir: Path, max_retries: int = 3) -> tuple[int, str, float, bool]:
        start_time = time.time()
        page_file = output_dir / f"page_{page_num}.md"

        if page_file.exists():
            with open(page_file, "r", encoding="utf-8") as f:
                return page_num, f.read(), time.time() - start_time, True

        with self._pdf_lock:
            image = self.loader.get_page(page_num)

        for attempt in range(max_retries):
            try:
                content_chunks = []
                for chunk in self.ocr_client.recognize_stream(self.system_prompt, self.prompt, image):
                    content_chunks.append(chunk)
                content = "".join(content_chunks)

                with open(page_file, "w", encoding="utf-8") as f:
                    f.write(content)

                return page_num, content, time.time() - start_time, False

            except requests.exceptions.RequestException:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    print(f"\n[警告] 第 {page_num} 页请求失败已达 {max_retries} 次，放弃并跳过该页。\n")
                    return page_num, "", time.time() - start_time, False

    def process(self, start_page: Optional[int] = None, end_page: Optional[int] = None, max_workers: int = 4):
        output_dir = self.pdf_path.parent / self.pdf_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        start = start_page if start_page is not None else 1
        end = end_page if end_page is not None else self.loader.total_pages

        results_dict = {}
        total_tasks = end - start + 1
        completed_tasks = 0
        actual_processed_count = 0

        print(f"开始处理，共计 {total_tasks} 页...")

        global_start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._process_page, page_num, output_dir)
                for page_num in range(start, end + 1)
            ]

            for future in concurrent.futures.as_completed(futures):
                page_num, content, elapsed, is_cached = future.result()
                results_dict[page_num] = content
                completed_tasks += 1

                if not is_cached:
                    actual_processed_count += 1
                    current_global_elapsed = time.time() - global_start_time
                    current_average = current_global_elapsed / actual_processed_count
                    print(
                        f"进度: {completed_tasks}/{total_tasks} (第 {page_num} 页已完成, "
                        f"线程阻塞耗时: {elapsed:.2f}秒, 实时平均耗时: {current_average:.2f} 秒/页)"
                    )

        global_elapsed_time = time.time() - global_start_time

        all_contents = [results_dict[i] for i in range(start, end + 1) if results_dict[i]]

        if start == 1 and end == self.loader.total_pages:
            final_md_path = self.pdf_path.parent / f"{self.pdf_path.stem}.md"
        else:
            final_md_path = self.pdf_path.parent / f"{self.pdf_path.stem}_{start}_to_{end}.md"

        with open(final_md_path, "w", encoding="utf-8") as f:
            f.write("\n\n---\n\n".join(all_contents))

        print(f"\n处理完成，合并后的文件保存在: {final_md_path}")
        print(f"总计真实处理: {actual_processed_count} 页 (跳过 {total_tasks - actual_processed_count} 页本地缓存)")
        print(f"总计真实耗时: {global_elapsed_time:.2f} 秒")


if __name__ == "__main__":
    API_KEY = "xxx"
    PDF_FILE = "中国佛教学术论典  4  《摩诃止观》之圆顿义  佛教的般若思想及其在中国的发展_13387092(1).pdf"

    SYSTEM_PROMPT = (
        "你是一个影印PDF转录Markdown文档的专家。"
        "你会把页底的注脚注释放到正文对应位置的后面。"
        "你会忽略页眉页脚，只关注有价值的部分。"
        "你只会输出识别的内容。"
    )
    PROMPT = "这是关于[佛法,摩诃止观]的竖版文档片段图。请将这张图片转录为很自然的Markdown格式。"

    max_workers = 64

    ocr_client = GeminiOCR(api_key=API_KEY)
    processor = PDFBatchProcessor(
        pdf_path=PDF_FILE,
        ocr_client=ocr_client,
        system_prompt=SYSTEM_PROMPT,
        prompt=PROMPT
    )

    processor.process(max_workers=max_workers)