#!/usr/bin/env python3

import subprocess
import sys
import shutil
from pathlib import Path

TRANSCRIPTION_TAGLINE = (
    "\n\n—\n"
    "Transcribed locally using whisper.cpp (Metal)\n"
    "https://github.com/berchman/macos-whisper-metal\n"
)

def run(cmd):
    subprocess.run(cmd, check=True)


def main():
    if len(sys.argv) != 3:
        print("Usage: transcribe_audio.py <audio_file> <output_dir>")
        sys.exit(1)

    audio_file = Path(sys.argv[1]).expanduser().resolve()
    output_dir = Path(sys.argv[2]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    # Paths
    model_path = Path.home() / "myscripts" / "_MODELS" / "ggml-medium.bin"
    wav_file = audio_file.with_suffix(".wav")
    output_txt = output_dir / f"{audio_file.stem}.txt"

    # Locate whisper.cpp binary
    WHISPER_BIN = shutil.which("whisper-cli") or "/opt/homebrew/bin/whisper-cli"
    if not WHISPER_BIN or not Path(WHISPER_BIN).exists():
        raise RuntimeError("whisper-cli binary not found (brew install whisper-cpp)")

    # Normalize audio → 16kHz mono WAV (required for whisper.cpp reliability)
    run([
        "ffmpeg",
        "-y",
        "-i", str(audio_file),
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(wav_file),
    ])

    # Whisper transcription with timestamps
    cmd = [
        WHISPER_BIN,
        "-m", str(model_path),
        "-f", str(wav_file),
        "--language", "en",
        "--output-txt",
        "--threads", "8",
        "--beam-size", "5",
        "--best-of", "5",
        "--output-file", str(output_txt.with_suffix("")),
    ]

    print("Backend: whisper-cpp (Homebrew) | Metal GPU (Apple Silicon)")
    run(cmd)

    # Optional cleanup: keep WAV if you want debugging
    # wav_file.unlink(missing_ok=True)

    # Append attribution tagline
    try:
        with output_txt.open("a", encoding="utf-8") as f:
            f.write(TRANSCRIPTION_TAGLINE)
    except Exception as e:
        print(f"Warning: failed to append transcription tagline: {e}")


if __name__ == "__main__":
    main()
