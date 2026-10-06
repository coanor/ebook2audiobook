# CosyVoice, Qwen3-TTS and IndexTTS with run.sh

Run these commands from the `chapter-resume` worktree. The source projects,
downloaded models, environments and book progress are shared with the main
checkout through `start-local.sh`.

```bash
bash run.sh --ebook "/path/book.epub" --tts=cosyvoice
bash run.sh --ebook "/path/book.epub" --tts=qwen3
bash run.sh --ebook "/path/book.epub" --tts=indextts
```

`--tts ENGINE` and the original `--tts_engine ENGINE` are also accepted.
Engine selection defaults to XTTS. The older `BOOK_FILE [TTS_ENGINE]` positional
syntax remains supported.

For a short sample, pass literal text (not a file path):

```bash
bash run.sh --tts=cosyvoice --text "五千年的文明，从这里开始。"
```

A text file is a file input: `bash run.sh --tts=qwen3 --ebook "/path/book.txt"`.
Choose exactly one of `--ebook` and `--text`. Raw-text audio defaults to the
caller’s current directory; file audio defaults to the source file’s directory.
Use `--output_dir /path/output` or `OUTPUT_DIR` to override it (the flag takes
priority). Other existing options, including `--chapter`, `--voice` and
`--speaker`, can be passed alongside these flags. Chapter selection and
automatic remembered resume apply to `--ebook`; raw text uses the existing
text-conversion flow.

Each engine exports completed book chapters beside the book, using the existing
chapter filename rules. Audio and subtitle filenames include the engine selected
with `--tts`, for example `置身事内_qwen3_chapter1_上篇_微观机制.m4b` and its
matching `.vtt`. Raw-text names use the text prefix and session ID followed by
the engine: `五千年的文明。_<session-id>_cosyvoice_chapter1_<title>.m4b`.
Chapter samples retain `_sample` before the engine name. Rerunning the same
command resumes automatically. Sentence cache paths are unchanged; older cached
audio can be exported again with the new names while existing output files stay
in place. Conversions already running keep their earlier names until restarted.
`--chapter "序言"` or `--chapter 3` converts a single chapter in a separate
sample session. `--output_format wav` changes the default M4B format.

List selectable entries, including chapters nested under a part, before choosing
a number. Numbers refer to this list, not the chapter numbers printed in the book.
Selecting a title includes its own subsections and stops before sibling chapters:

```bash
bash run.sh --ebook "/mnt/d/ai/books/如何阅读一本书.epub" --tts=cosyvoice --list_chapters
bash run.sh --ebook "/mnt/d/ai/books/如何阅读一本书.epub" --tts=cosyvoice --chapter "第四章 阅读的第二个层次：检视阅读"
bash run.sh --ebook "/mnt/d/ai/books/置身事内.epub" --tts=qwen3 --chapter "第四章 工业化中的政府角色"
```

Nested selections use a separate remembered session from older part samples.
Existing full-book conversions keep their top-level chapter grouping.

## Qwen3 batching

Qwen3 generates up to four sentences together on CUDA by default, or one on CPU.
Set `--batch_size` (1–16) to tune it; `1` restores single-sentence generation:

```bash
env -u CUDA_LAUNCH_BLOCKING bash run.sh --tts=qwen3 \
  --ebook "/mnt/d/ai/books/置身事内.epub" \
  --chapter "第四章 工业化中的政府角色" --batch_size 4
```

Batches stay within each text block. Every sentence keeps its own atomic FLAC
cache and subtitle entry, including pause tags. Completed chapters still export
immediately. Changing batch size keeps the same remembered session and cached
audio; files completed ahead of a saved progress index are reused after interruption.
The log reports elapsed time, audio duration and RTF per batch: RTF below 1 means
generation was faster than playback. Performance depends on sentence lengths and
available GPU memory; if a batch runs out of memory, retry with `--batch_size 2`
or `1`. Existing running processes use their already loaded code until restarted.

## Voices

Qwen3 uses the downloaded **1.7B CustomVoice** model and defaults to **Uncle_Fu**,
a mature Chinese male preset. Choose another preset with `--speaker`:

```bash
bash run.sh --ebook "/path/book.epub" --tts=qwen3 --speaker Vivian --chapter "序言"
```

Supported presets: `Uncle_Fu`, `Vivian`, `Serena`, `Dylan`, `Eric`, `Ryan`,
`Aiden`, `Ono_Anna`, `Sohee`. Names are case insensitive. This CustomVoice model
does not clone recordings; `--voice` and `--voice_map` are rejected for Qwen3.
Changing the speaker selects a separate resume session.

CosyVoice and IndexTTS accept a reference recording:

```bash
bash run.sh --ebook "/path/book.epub" --tts=cosyvoice --voice "/path/reference.wav"
bash run.sh --ebook "/path/book.epub" --tts=indextts --voice "/path/reference.wav"
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
