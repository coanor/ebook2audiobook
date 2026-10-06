#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo 'Usage: bash run.sh --ebook BOOK_FILE [--tts=ENGINE] [additional options]'
  echo '       bash run.sh --text "Raw text" [--tts=ENGINE] [additional options]'
  echo 'ENGINE defaults to xtts. --tts ENGINE and --tts_engine ENGINE also work.'
  echo 'Available: xtts, cosyvoice, qwen3, indextts, bark, piper, tortoise, vits, fairseq, glowtts, tacotron, yourtts.'
  echo 'Qwen3 uses --speaker Uncle_Fu by default; CosyVoice and IndexTTS accept --voice reference.wav.'
  echo 'Qwen3 batches 4 sentences on CUDA (1 on CPU); set --batch_size 1..16 to change it.'
  echo 'Output defaults to the book directory, or the current directory for raw text.'
  echo 'Use --output_dir DIR or OUTPUT_DIR to choose another output directory.'
  echo 'Each completed book chapter is exported immediately; rerun the same command to resume.'
  echo 'Example: bash run.sh --tts=qwen3 --ebook "/mnt/d/ai/books/jinrong.epub" --chapter "序言"'
  echo 'Example: bash run.sh --tts=cosyvoice --text "五千年的文明。"'
  echo 'Legacy BOOK_FILE [TTS_ENGINE] arguments remain supported.'
}

fail() { printf '%s\n' "$1" >&2; exit 2; }
require_value() {
  [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || fail "Missing value for $1 (use $1=VALUE for a value starting with '-')."
}
set_input() {
  [[ -z "$input_kind" ]] || fail 'Choose exactly one input: --ebook or --text.'
  [[ -n "$2" ]] || fail "Empty value for --$1."
  input_kind="$1"
  input_value="$2"
}

[[ $# -gt 0 ]] || { usage; exit 2; }

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
input_kind=''
input_value=''
tts_engine=xtts
output_dir="${OUTPUT_DIR:-}"
extra_args=()

# Recognize legacy positional arguments only at the beginning; values of
# forwarded options (e.g. --chapter TITLE) must never become input files.
if [[ "$1" != -* ]]; then
  set_input ebook "$1"
  shift
  if [[ $# -gt 0 && "$1" != -* ]]; then
    tts_engine="$1"
    shift
  fi
fi
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --ebook|--text)
      require_value "$@"
      set_input "${1#--}" "$2"
      shift 2 ;;
    --ebook=*|--text=*)
      option="${1%%=*}"
      set_input "${option#--}" "${1#*=}"
      shift ;;
    --tts|--tts_engine)
      require_value "$@"
      tts_engine="$2"
      shift 2 ;;
    --tts=*|--tts_engine=*)
      tts_engine="${1#*=}"
      shift ;;
    --output_dir)
      require_value "$@"
      output_dir="$2"
      shift 2 ;;
    --output_dir=*)
      output_dir="${1#*=}"
      [[ -n "$output_dir" ]] || fail 'Empty value for --output_dir.'
      shift ;;
    *) extra_args+=("$1"); shift ;;
  esac
done
[[ -n "$input_kind" ]] || fail 'Provide --ebook BOOK_FILE or --text "Raw text".'
tts_engine="${tts_engine,,}"
case "$tts_engine" in
  xtts|cosyvoice|qwen3|indextts|bark|piper|tortoise|vits|fairseq|glowtts|tacotron|yourtts) ;;
  *)
    printf 'Unsupported TTS engine: %s\n' "$tts_engine" >&2
    echo 'Use bash run.sh --help to see available engines.' >&2
    exit 2
    ;;
esac

input_args=()
default_output_dir="$PWD"
if [[ "$input_kind" == ebook ]]; then
  [[ -f "$input_value" ]] || fail "Book file does not exist: $input_value"
  book_dir="$(cd -- "$(dirname -- "$input_value")" && pwd)"
  input_args=(--ebook "$book_dir/$(basename -- "$input_value")" --split_by_chapter)
  default_output_dir="$book_dir"
else
  input_args=(--text "$input_value")
  if [[ "$input_value" == -* ]]; then
    input_args=("--text=$input_value")
  fi
fi
output_dir="${output_dir:-$default_output_dir}"
mkdir -p -- "$output_dir"
output_dir="$(cd -- "$output_dir" && pwd)"

exec "$repo_dir/start-local.sh" --headless \
  "${input_args[@]}" \
  --language zho \
  --device cuda \
  --tts_engine "$tts_engine" \
  --output_format m4b \
  --output_dir "$output_dir" \
  "${extra_args[@]}"

# Additional options: --voice "/path/to/reference.wav", --new_session,
# --output_format wav, --speaker Vivian (qwen3), or --tts_model_dir /path/model.
