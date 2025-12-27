#!/usr/bin/env python3
"""
transcribe_audio.py

Usage:
  python3 transcribe_audio.py <audio_file> <output_dir>

What it does:
- Converts input audio to 16kHz mono WAV (temp) using ffmpeg (then deletes it).
- Runs whisper.cpp (Homebrew whisper-cli) using Metal GPU on Apple Silicon.
- Produces a single .txt transcript with:
    - timestamps per segment
    - conservative paragraph breaks (gap >= 1.5s)
    - <h6>Word count footer</h6> (included in transcript AND WordPress)
    - a provenance/tagline + checksum block at the very end (transcript-only)
      (and will not double-append on re-runs)
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# --------- user-tunable defaults ----------
DEFAULT_MODEL = Path.home() / "myscripts" / "_MODELS" / "ggml-medium.bin"
DEFAULT_LANGUAGE = "en"
DEFAULT_THREADS = "8"
DEFAULT_BEAM = "5"
DEFAULT_BEST_OF = "5"
PARA_BREAK_SECONDS = 1.5

# Put your public repo URL here:
PROJECT_URL = "https://github.com/berchman/macos-whisper-metal"
TAGLINE_MARKER = "Transcribed locally with whisper.cpp (Metal) on macOS."
CHECKSUM_MARKER = "Checksum:"
# ----------------------------------------

VTT_CUE_RE = re.compile(
    r"(?P<start>\d{2}:\d{2}:\d{2}\.\d{3})\s+-->\s+(?P<end>\d{2}:\d{2}:\d{2}\.\d{3})"
)

def which_or_die(name: str, hint: str) -> str:
    p = shutil.which(name)
    if not p:
        raise RuntimeError(f"Missing required binary: {name}\nHint: {hint}")
    return p

def hms_ms_to_seconds(ts: str) -> float:
    # "HH:MM:SS.mmm"
    hh, mm, rest = ts.split(":")
    ss, ms = rest.split(".")
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + (int(ms) / 1000.0)

def seconds_to_mmss(seconds: float) -> str:
    s = max(0.0, seconds)
    mm = int(s // 60)
    ss = int(s % 60)
    return f"{mm:02d}:{ss:02d}"

@dataclass
class Cue:
    start_s: float
    end_s: float
    text: str

def parse_vtt(vtt_text: str) -> List[Cue]:
    lines = [ln.rstrip("\n") for ln in vtt_text.splitlines()]
    cues: List[Cue] = []

    i = 0
    # Skip header lines until we find cues
    while i < len(lines):
        ln = lines[i].strip()
        m = VTT_CUE_RE.search(ln)
        if m:
            start_s = hms_ms_to_seconds(m.group("start"))
            end_s = hms_ms_to_seconds(m.group("end"))
            i += 1
            buf: List[str] = []
            while i < len(lines) and lines[i].strip() != "":
                buf.append(lines[i].strip())
                i += 1
            text = " ".join(buf).strip()
            if text:
                cues.append(Cue(start_s=start_s, end_s=end_s, text=text))
        i += 1

    # Basic de-dupe / cleanup
    cleaned: List[Cue] = []
    prev = None
    for c in cues:
        if prev and c.text == prev.text and abs(c.start_s - prev.start_s) < 0.01:
            continue
        cleaned.append(c)
        prev = c
    return cleaned

def build_timestamped_paragraphs(cues: List[Cue], para_break_s: float) -> str:
    if not cues:
        return ""

    out_lines: List[str] = []
    prev_end = cues[0].start_s

    for idx, c in enumerate(cues):
        gap = c.start_s - prev_end
        if idx > 0 and gap >= para_break_s:
            out_lines.append("")  # blank line = paragraph break

        stamp = seconds_to_mmss(c.start_s)
        out_lines.append(f"[{stamp}] {c.text}")
        prev_end = c.end_s

    return "\n".join(out_lines).strip()

def strip_timestamps_for_wordcount(text: str) -> str:
    return re.sub(r"^\[\d{2}:\d{2}\]\s*", "", text, flags=re.M)

def word_count(text: str) -> int:
    words = re.findall(r"\b[\w']+\b", text)
    return len(words)

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()

def already_has_footer(transcript_text: str) -> bool:
    return (TAGLINE_MARKER in transcript_text) and (CHECKSUM_MARKER in transcript_text)

def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: transcribe_audio.py <audio_file> <output_dir>")
        sys.exit(1)

    audio_file = Path(sys.argv[1]).expanduser().resolve()
    output_dir = Path(sys.argv[2]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    model_path = Path(os.environ.get("WHISPER_MODEL", str(DEFAULT_MODEL))).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    ffmpeg = which_or_die("ffmpeg", "Install with: brew install ffmpeg")
    whisper_cli = shutil.which("whisper-cli") or shutil.which("whisper-command") or ""
    if not whisper_cli:
        # Homebrew installs whisper-cli under its bin, but PATH may differ under LaunchAgents.
        # Try common Homebrew locations:
        for p in [
            "/opt/homebrew/bin/whisper-cli",
            "/usr/local/bin/whisper-cli",
        ]:
            if Path(p).exists():
                whisper_cli = p
                break
    if not whisper_cli:
        raise RuntimeError("whisper-cli not found. Install with: brew install whisper-cpp")

    out_txt = output_dir / f"{audio_file.stem}.txt"

    # If re-running on an existing transcript, avoid re-appending footer blocks.
    existing = ""
    if out_txt.exists():
        existing = out_txt.read_text(encoding="utf-8", errors="ignore")

    # Convert to temp WAV (16k mono)
    with tempfile.TemporaryDirectory(prefix="whisper_transcribe_") as td:
        td_path = Path(td)
        wav_path = td_path / f"{audio_file.stem}.wav"
        base_out = td_path / f"{audio_file.stem}"

        # Convert with ffmpeg
        conv_cmd = [
            ffmpeg,
            "-y",
            "-i", str(audio_file),
            "-ac", "1",
            "-ar", "16000",
            "-c:a", "pcm_s16le",
            str(wav_path),
        ]
        subprocess.run(conv_cmd, check=True)

        # Run whisper-cli, output VTT (we will format our own TXT)
        cmd = [
            whisper_cli,
            "-m", str(model_path),
            "-f", str(wav_path),
            "--language", DEFAULT_LANGUAGE,
            "--threads", DEFAULT_THREADS,
            "--beam-size", DEFAULT_BEAM,
            "--best-of", DEFAULT_BEST_OF,
            "--output-vtt",
            "--output-file", str(base_out),
        ]

        print("Backend: whisper-cpp (Homebrew) | Metal GPU (Apple Silicon)")
        subprocess.run(cmd, check=True)

        vtt_path = base_out.with_suffix(".vtt")
        if not vtt_path.exists():
            raise RuntimeError("Expected .vtt output not found. whisper-cli may have changed output behavior.")

        cues = parse_vtt(vtt_path.read_text(encoding="utf-8", errors="ignore"))
        transcript_body = build_timestamped_paragraphs(cues, PARA_BREAK_SECONDS)

        # Word count based on text without timestamps
        wc = word_count(strip_timestamps_for_wordcount(transcript_body))
        wc_footer = f"<h6>Word count: {wc}</h6>"

        # Checksum should represent the "meaningful content" portion (no footer/tagline)
        digest = sha256_text(strip_timestamps_for_wordcount(transcript_body).strip())

        tagline_block = "\n".join([
            "",
            "---",
            TAGLINE_MARKER,
            f"Project: {PROJECT_URL}",
            f"{CHECKSUM_MARKER} {digest}",
        ]).strip("\n")

        final_text = transcript_body.strip() + "\n\n" + wc_footer + "\n" + tagline_block + "\n"

        # If an older transcript already exists, don't double-append: just overwrite cleanly.
        # This is the simplest “checksum-safe” behavior.
        out_txt.write_text(final_text, encoding="utf-8")

    # Done. Temp WAV automatically deleted via TemporaryDirectory.

if __name__ == "__main__":
    main()
