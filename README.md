# macOS Whisper Transcription (Metal)

A production-stable, automated audio transcription pipeline for **Apple Silicon Macs**
using **whisper-cpp with native Metal GPU acceleration**.

This project intentionally avoids Python ML inference stacks in favor of a
deterministic, native backend that is fast, stable, and low-maintenance on macOS.

---

## Features

- 📂 Watch-folder based automation
- ⚡ Apple Metal GPU acceleration (M1 / M2 / M3)
- 🕒 Timestamps enabled
- 🧑‍🤝‍🧑 Speaker diarization (experimental)
- 🚀 Runs automatically via macOS LaunchAgent
- 🧊 System “freeze” documentation for long-term stability

---

## Why whisper-cpp (not PyTorch)

On Apple Silicon, Python GPU backends can be fragile.
This pipeline uses **whisper-cpp** to achieve:

- Zero PyTorch / CUDA / MPS dependency churn
- Deterministic inference
- Predictable performance
- Minimal runtime surface area

Python is used **only for orchestration**, not model execution.

---

## Requirements

- macOS (Apple Silicon)
- Homebrew
- `whisper-cpp`
- `ffmpeg`
- `fswatch`

---

## Installation

```bash
brew install whisper-cpp ffmpeg fswatch
```
## Download a model:
```bash
mkdir -p _MODELS
curl -L -o _MODELS/ggml-medium.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin
```  
## Usage
```bash
./02_transcribe_audio_v3.sh
```
Drop audio files into 
```bash
_00_To_Be_Processed/.
```
Transcripts will appear in:
```bash
__02_Transcripts/
```

## Automation

The watcher can be run automatically on login via a macOS LaunchAgent.
See SYSTEM_FREEZE.md for details.


## Stability Notes

This system is intentionally conservative.

Before upgrading:
	•	Read SYSTEM_FREEZE.md
	•	Test changes in isolation
	•	Avoid blind brew upgrade


## License

MIT

```bash
This tone signals:
- competence
- restraint
- real-world experience

Exactly what you want for a public repo.

---

## Phase 3 — Badges + demo GIF

### Badges (simple, tasteful)

Add this **directly under the title** in `README.md`:

```md
![macOS](https://img.shields.io/badge/macOS-Apple%20Silicon-black)
![Metal](https://img.shields.io/badge/GPU-Metal-blue)
![License](https://img.shields.io/badge/license-MIT-green)
```

