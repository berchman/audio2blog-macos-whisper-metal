import sys
import os
import subprocess

def main():
    if len(sys.argv) < 3:
        print("Usage: python transcribe_audio.py <audio_file> <output_dir>")
        sys.exit(1)

    audio_file = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.exists(audio_file):
        print(f"File not found: {audio_file}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(audio_file))[0]
    out_path = os.path.join(output_dir, base)  # whisper-cli appends .txt when using --output-txt

    BIN = "/opt/homebrew/bin/whisper-cli"
    MODEL = os.path.expanduser("~/myscripts/_MODELS/ggml-medium.bin")

    if not os.path.exists(BIN):
        raise RuntimeError(f"whisper-cli not found at {BIN}. Run: brew install whisper-cpp")
    if not os.path.exists(MODEL):
        raise RuntimeError(f"Model not found at {MODEL}. Download ggml-medium.bin into ~/myscripts/_MODELS/")
    
    cmd = [
        BIN,
        "--model", MODEL,
        "--file", audio_file,
        "--output-txt",
        "--output-file", out_path,
        "--language", "en",
        "--threads", "8",
        "--diarize",
        "--diarize-min-speakers", "1",
        "--diarize-max-speakers", "2",
    ]  

    print("Backend: whisper-cpp (Homebrew) | Metal GPU (Apple Silicon)")
    subprocess.run(cmd, check=True)

    txt_path = out_path + ".txt"
    print(f"Transcript written to: {txt_path}")

if __name__ == "__main__":
    main()