#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 || "$1" == "--help" || "$1" == "-h" ]]; then
  echo 'Usage: bash run.sh BOOK_FILE [TTS_ENGINE] [additional options]'
  echo 'TTS_ENGINE defaults to xtts. You can also pass --tts_engine ENGINE.'
  echo 'Available: xtts, cosyvoice, qwen3, indextts, bark, piper, tortoise, vits, fairseq, glowtts, tacotron, yourtts.'
  echo 'Qwen3 uses --speaker Uncle_Fu by default; CosyVoice and IndexTTS accept --voice reference.wav.'
  echo 'Output defaults to the book directory. Set OUTPUT_DIR to use a shared folder.'
  echo 'Each completed book chapter is exported immediately; rerun the same command to resume.'
  echo 'Example: bash run.sh "/mnt/d/ai/books/jinrong.epub"'
  echo 'Example: bash run.sh "/mnt/d/ai/books/jinrong.epub" bark --chapter "序言"'
  echo 'Example: bash run.sh "/mnt/d/ai/books/jinrong.epub" qwen3 --speaker Uncle_Fu --chapter "序言"'
  if [[ $# -eq 0 ]]; then
    exit 2
  fi
  exit 0
fi

if [[ ! -f "$1" ]]; then
  printf 'Book file does not exist: %s\n' "$1" >&2
  exit 1
fi

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
book_dir="$(cd -- "$(dirname -- "$1")" && pwd)"
ebook_path="$book_dir/$(basename -- "$1")"
shift

tts_engine=xtts
if [[ $# -gt 0 && "$1" != -* ]]; then
  tts_engine="${1,,}"
  shift
fi
case "$tts_engine" in
  xtts|cosyvoice|qwen3|indextts|bark|piper|tortoise|vits|fairseq|glowtts|tacotron|yourtts) ;;
  *)
    printf 'Unsupported TTS engine: %s\n' "$tts_engine" >&2
    echo 'Use bash run.sh --help to see available engines.' >&2
    exit 2
    ;;
esac

output_dir="${OUTPUT_DIR:-$book_dir}"
mkdir -p -- "$output_dir"
output_dir="$(cd -- "$output_dir" && pwd)"

exec "$repo_dir/start-local.sh" --headless \
  --ebook "$ebook_path" \
  --language zho \
  --device cuda \
  --tts_engine "$tts_engine" \
  --output_format m4b \
  --split_by_chapter \
  --output_dir "$output_dir" \
  "$@"

# Additional options: --voice "/path/to/reference.wav", --new_session,
# --output_format wav, --speaker Vivian (qwen3), or --tts_model_dir /path/model.
