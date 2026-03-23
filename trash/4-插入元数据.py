import os
from pydub import AudioSegment


class AudioMetadataManager:
    def __init__(self, audio_path, md_path, output_md_path):
        self.audio_path = audio_path
        self.md_path = md_path
        self.output_md_path = output_md_path

    def _get_audio_info(self):
        title = os.path.splitext(os.path.basename(self.audio_path))[0]
        audio = AudioSegment.from_mp3(self.audio_path)
        duration_seconds = int(audio.duration_seconds)
        hours, remainder = divmod(duration_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return title, duration_str

    def _get_user_input(self):
        year = input("请输入音频的创建时间（年份）: ")
        author = input("请输入音频的作者: ")
        original_text = input("请输入音频的原文: ")
        return year, author, original_text

    def _generate_metadata_string(self, title, duration, year, author, original_text):
        return (
            f"> 标题：{title}\n"
            f"> 时间：{year}\n"
            f"> 时长：{duration}\n"
            f"> 作者：{author}\n"
            f"> 原文：{original_text}\n"
            "\n"
            "---\n\n"
        )

    def process(self):
        title, duration = self._get_audio_info()
        year, author, original_text = self._get_user_input()

        metadata = self._generate_metadata_string(
            title, duration, year, author, original_text
        )

        with open(self.md_path, "r", encoding="utf-8") as f:
            content = f.read()

        with open(self.output_md_path, "w", encoding="utf-8") as f:
            f.write(metadata + content)

        print(f"[{title}] 的元数据已成功写入到新的文件：{self.output_md_path}")


if __name__ == "__main__":
    # 音频路径
    audio_file = "摩诃止观-久仁法师/摩诃止观016/摩诃止观016.mp3"
    # 书面稿路径
    md_file = "摩诃止观-久仁法师/摩诃止观016/摩诃止观016_书面稿.md"

    base_name, ext = os.path.splitext(md_file)
    output_file = f"{base_name}_元数据{ext}"

    manager = AudioMetadataManager(audio_file, md_file, output_file)
    manager.process()