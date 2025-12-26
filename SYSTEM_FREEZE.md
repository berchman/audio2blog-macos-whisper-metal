# Transcription System — Frozen State

## Purpose
Automated audio transcription using whisper.cpp with Apple Metal GPU acceleration.

This system is intentionally stable and should not be casually upgraded.

---

## Core Components

- macOS Apple Silicon (M1 Pro)
- Homebrew whisper-cpp (Metal enabled)
- Model: ggml-medium.bin
- Python used ONLY as orchestration (no ML inference)
- Folder watcher via fswatch
- Auto-run via LaunchAgent

---

## Important Rules

DO NOT:
- `brew upgrade` blindly
- Pull whisper.cpp from git
- Change Python versions unnecessarily
- Replace the model without testing

DO:
- Treat this system as production
- Make changes intentionally
- Test upgrades in isolation

---

## File Layout

## myscripts/
## ├── _00_To_Be_Processed/
## ├── _01_Processed/
## ├── _02_Transcripts/
## ├── _MODELS/
## │   └── ggml-medium.bin
## ├── _LOGS/
## ├── transcribe_audio.py
## ├── 02_transcribe_audio_v3.sh
## └── SYSTEM_FREEZE.md
##

---

## LaunchAgent

Label: `com.bert.transcribe`

Auto-runs:
`02_transcribe_audio_v3.sh`

Logs:
- `_LOGS/launchagent.out`
- `_LOGS/launchagent.err`

---

## Last Known Good State
Date: 2025-12-14  
Status: Fully working, GPU accelerated, stable

03_publish_to_wordpress.py frozen at v1.0
Voice-gated, timestamp-safe, idempotent WordPress draft publisher
