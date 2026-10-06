# CosyVoice, Qwen3-TTS and IndexTTS with run.sh

Run these commands from the `chapter-resume` worktree. The source projects,
downloaded models, environments and book progress are shared with the main
checkout through `start-local.sh`.

```bash
bash run.sh "/path/book.epub" cosyvoice
bash run.sh "/path/book.epub" qwen3
bash run.sh "/path/book.epub" indextts
```

Each engine exports completed book chapters beside the book, using the existing
chapter filename rules. Rerunning the same command resumes automatically.
`--chapter "序言"` or `--chapter 3` converts a single chapter in a separate
sample session. `--output_format wav` changes the default M4B format.

## Voices

Qwen3 uses the downloaded **1.7B CustomVoice** model and defaults to **Uncle_Fu**,
a mature Chinese male preset. Choose another preset with `--speaker`:

```bash
bash run.sh "/path/book.epub" qwen3 --speaker Vivian --chapter "序言"
```

Supported presets: `Uncle_Fu`, `Vivian`, `Serena`, `Dylan`, `Eric`, `Ryan`,
`Aiden`, `Ono_Anna`, `Sohee`. Names are case insensitive. This CustomVoice model
does not clone recordings; `--voice` and `--voice_map` are rejected for Qwen3.
Changing the speaker selects a separate resume session.

CosyVoice and IndexTTS accept a reference recording:

```bash
bash run.sh "/path/book.epub" cosyvoice --voice "/path/reference.wav"
bash run.sh "/path/book.epub" indextts --voice "/path/reference.wav"
```

Without `--voice`, both use CosyVoice's bundled Chinese example recording at
`components/external-tts/CosyVoice/asset/zero_shot_prompt.wav`.
CosyVoice uses cross-lingual cloning, which does not require a transcript of
the recording. IndexTTS uses the 2.5 multilingual inference implementation.

## Environments and model locations

Each engine runs in an isolated, persistent Python worker. Its model loads once
per book conversion; the existing conversion process keeps handling sentence
audio, FLAC caches, subtitles and chapter exports. Workers stop when conversion
finishes, fails or is interrupted.

Models default to these shared folders:

- `models/tts/external/CosyVoice3`
- `models/tts/external/Qwen3-TTS-1.7B-CustomVoice`
- `models/tts/external/IndexTTS-2.5` (including `hf_cache` auxiliary models)

Use `--tts_model_dir /path/model` to override a model directory. The supported
variants remain CosyVoice 3, Qwen3 CustomVoice and IndexTTS 2.5. A model-directory
override also selects a separate resume session. `--custom_model` ZIP archives
and other `--fine_tuned` presets are not supported by these adapters.

Environments are installed under `run/external-tts/{engine}`. To reinstall or
prepare another machine with the same downloaded projects:

```bash
bash setup-external-tts.sh all
# Or install one engine:
bash setup-external-tts.sh qwen3
```

Setup uses Python 3.11 and CUDA 12.8 PyTorch 2.8 in each environment, with separate
Transformers versions required by the projects. It installs inference dependencies
and checks imports and a small CUDA operation; it does not load TTS weights, start
a server or generate speech. TensorRT, FlashAttention and DeepSpeed are optional
and are not enabled. CosyVoice uses a newer Diffusers/ONNX Runtime pair compatible
with these dependencies; IndexTTS uses OpenCV 4.11 for its NumPy 2.x dependency.
If SoX is missing, setup downloads an Ubuntu binary into the shared runtime and
exposes it only inside the engine environments, without a system-wide install.

`--device cpu` is supported for these engines. The prepared setup targets the
local NVIDIA GPU; other device backends are not supported by these adapters.
