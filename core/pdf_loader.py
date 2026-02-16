from pathlib import Path
from pdf2image import convert_from_path, pdfinfo_from_path
from PIL import Image


class PDFLoader:
    # 加载pdf的某一页
    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"找不到文件: {self.pdf_path}")

        info = pdfinfo_from_path(str(self.pdf_path))
        self.total_pages = int(info["Pages"])

    def get_page(self, page_num: int, dpi: int = 300) -> Image.Image:
        if page_num < 1 or page_num > self.total_pages:
            raise ValueError(f"页码超出范围: {page_num} (共 {self.total_pages} 页)")

        images = convert_from_path(
            str(self.pdf_path),
            dpi=dpi,
            first_page=page_num,
            last_page=page_num
        )
        return images[0]


if __name__ == "__main__":
    loader = PDFLoader("摩诃止观选读.pdf")
    print(f"PDF 加载完成，共 {loader.total_pages} 页。")
    page_img = loader.get_page(1)
    print(f"第 1 页尺寸: {page_img.size}")
    page_img.show()