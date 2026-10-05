#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 || "$1" == "--help" || "$1" == "-h" ]]; then
  echo 'Usage: bash run.sh BOOK_FILE [additional options]'
  echo 'Output defaults to the book directory. Set OUTPUT_DIR to use a shared folder.'
  echo 'Each completed book chapter is exported immediately; rerun the same command to resume.'
  echo 'Example: bash run.sh "/mnt/d/ai/books/jinrong.epub"'
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

output_dir="${OUTPUT_DIR:-$book_dir}"
mkdir -p -- "$output_dir"
output_dir="$(cd -- "$output_dir" && pwd)"

exec "$repo_dir/start-local.sh" --headless \
  --ebook "$ebook_path" \
  --language zho \
  --device cuda \
  --tts_engine xtts \
  --output_format m4b \
  --split_by_chapter \
  --output_dir "$output_dir" \
  "$@"

# Additional options: --voice "/path/to/reference.wav", --new_session,
# --output_format wav, or --tts_engine bark / piper.
