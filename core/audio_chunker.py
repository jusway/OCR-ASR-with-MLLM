import os
import time
import uuid
import glob
import subprocess
from pathlib import Path
from pydub import AudioSegment
from google import genai
from google.genai import types


class AudioChunker:
    def __init__(self, chunk_minutes=3, overlap_seconds=10):
        self.chunk_ms = chunk_minutes * 60 * 1000
        self.overlap_ms = overlap_seconds * 1000
        self.step = self.chunk_ms - self.overlap_ms

    def _preprocess_audio(self, input_path: str) -> str:
        ext = Path(input_path).suffix
        output_path = f"temp_processed_full_{uuid.uuid4().hex}{ext}"

        command = [
            'ffmpeg',
            '-y',
            '-i', input_path,
            '-af', 'dynaudnorm,lowpass=f=8000',
            output_path
        ]

        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return output_path

    def process(self, file_path, output_dir="output_chunks"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            existing_files = sorted(glob.glob(os.path.join(output_dir, "chunk_*.mp3")))
            if existing_files:
                return existing_files

        processed_full_audio_path = self._preprocess_audio(file_path)

        audio = AudioSegment.from_file(processed_full_audio_path)
        total_duration = len(audio)

        generated_files = []
        start = 0
        index = 0

        while start < total_duration:
            end = min(start + self.chunk_ms, total_duration)
            chunk = audio[start:end]

            out_path = os.path.join(output_dir, f"chunk_{index:03d}.mp3")
            chunk.export(out_path, format="mp3")
            generated_files.append(out_path)

            start += self.step
            index += 1

        os.remove(processed_full_audio_path)

        return generated_files