# SYSTEM_FREEZE (Beta 0.5)

**Status:** Beta 0.5 (timestamps deferred)

## Core invariants
- Local processing (no cloud inference)
- Native Metal backend via whisper.cpp (Homebrew `whisper-cli`)
- Python is glue only (conversion, parsing, publishing)
- Footer/tagline included in transcript, but **publisher strips it** by default
- Checksum prevents double-processing

## Folder layout (expected)
```
myscripts/
  _00_To_Be_Processed/
  _01_Processed/
  _02_Transcripts/
  _MODELS/
  _LOGS/
  02_transcribe_audio_v3.sh
  transcribe_audio.py
  03_publish_to_wordpress.py
```

## Deferred
- Timestamps (will be reintroduced as an explicit mode for scripts, not raw transcripts)
- Diarization (separate track; will likely use a dedicated tool, not whisper.cpp itself)
