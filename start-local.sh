#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
user_home="${HOME:?HOME is not set}"
calibre_dir="$user_home/.local/calibre-bin/calibre"
calibre_lib_dir="$user_home/.local/calibre-deps/usr/lib/x86_64-linux-gnu"

export PATH="$user_home/.local/bin:$calibre_dir:$PATH"
if [[ -d "$calibre_lib_dir" ]]; then
  export LD_LIBRARY_PATH="$calibre_lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

cd "$repo_dir"
exec "$repo_dir/python_env/bin/python" -u "$repo_dir/app.py" --script_mode native "$@"
