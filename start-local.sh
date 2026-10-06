#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
runtime_dir="${TXT2VOICE_RUNTIME_DIR:-$repo_dir}"

# Linked worktrees contain code but do not include the installed environment
# or downloaded models. Reuse the main checkout's runtime and saved sessions.
if [[ -z "${TXT2VOICE_RUNTIME_DIR:-}" && ! -x "$runtime_dir/python_env/bin/python" ]]; then
  if git_common_dir="$(git -C "$repo_dir" rev-parse --path-format=absolute --git-common-dir 2>/dev/null)"; then
    runtime_dir="$(dirname -- "$git_common_dir")"
  fi
fi
if [[ ! -x "$runtime_dir/python_env/bin/python" ]]; then
  printf 'Prepared Python environment not found in: %s\n' "$runtime_dir" >&2
  echo 'Set TXT2VOICE_RUNTIME_DIR to the checkout with the installed environment.' >&2
  exit 1
fi
runtime_dir="$(cd -- "$runtime_dir" && pwd)"
export TXT2VOICE_RUNTIME_DIR="$runtime_dir"

user_home="${HOME:?HOME is not set}"
calibre_dir="$user_home/.local/calibre-bin/calibre"
calibre_lib_dir="$user_home/.local/calibre-deps/usr/lib/x86_64-linux-gnu"

export PATH="$user_home/.local/bin:$calibre_dir:$PATH"
if [[ -d "$calibre_lib_dir" ]]; then
  export LD_LIBRARY_PATH="$calibre_lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

if [[ "$runtime_dir" != "$repo_dir" ]]; then
  printf 'Using shared runtime: %s\n' "$runtime_dir"
fi
cd "$runtime_dir"
exec "$runtime_dir/python_env/bin/python" -u "$repo_dir/app.py" --script_mode native "$@"
