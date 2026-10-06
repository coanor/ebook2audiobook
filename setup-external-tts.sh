#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
runtime_dir="${TXT2VOICE_RUNTIME_DIR:-$repo_dir}"
if [[ -z "${TXT2VOICE_RUNTIME_DIR:-}" && ! -x "$runtime_dir/python_env/bin/python" ]]; then
  git_common_dir="$(git -C "$repo_dir" rev-parse --path-format=absolute --git-common-dir)"
  runtime_dir="$(dirname -- "$git_common_dir")"
fi
exec "$runtime_dir/python_env/bin/python" "$repo_dir/tools/setup_external_tts.py" \
  --runtime-dir "$runtime_dir" "$@"
