#!/bin/bash

# Configuration
INPUT_DIR="$HOME/Desktop/convert_videos"
OUTPUT_DIR="$HOME/Desktop/converted_audio"
TRANSCRIPT_DIR="$HOME/Desktop/transcripts"
LOG_FILE="$HOME/Desktop/conversion_log.txt"

mkdir -p "$INPUT_DIR" "$OUTPUT_DIR" "$TRANSCRIPT_DIR"

convert_file() {
    input_file="$1"
    filename=$(basename "$input_file")
    filename_no_ext="${filename%.*}"
    output_file="$OUTPUT_DIR/${filename_no_ext}.mp3"
    transcript_file="$TRANSCRIPT_DIR/${filename_no_ext}.txt"
    
    echo "$(date): Processing $filename" >> "$LOG_FILE"
    
    # Convert to MP3 if needed
    if [[ "$input_file" =~ \.mp4$ ]]; then
        ffmpeg -i "$input_file" -vn -acodec libmp3lame -q:a 2 "$output_file" 2>> "$LOG_FILE"
        audio_file="$output_file"
    else
        cp "$input_file" "$output_file"
        audio_file="$input_file"
    fi
    
    if [ -f "$output_file" ]; then
        if whisper "$audio_file" --model base --output_dir "$TRANSCRIPT_DIR" --output_format txt 2>> "$LOG_FILE"; then
            mv "$TRANSCRIPT_DIR/${filename_no_ext}.mp3.txt" "$transcript_file"
            rm "$input_file"
            echo "$(date): Successfully processed $filename" >> "$LOG_FILE"
        fi
    fi
}

watch_directory() {
    echo "Watching $INPUT_DIR for MP4 and MP3 files..."
    
    fswatch -0 "$INPUT_DIR" | while read -d "" event
    do
        if [[ "$event" =~ \.(mp4|mp3)$ ]]; then
            sleep 1
            convert_file "$event"
        fi
    done
}

watch_directory
