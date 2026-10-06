"""Run a downloaded TTS project in its own environment; stdout is JSON only."""
import argparse
import importlib
import json
import os
import re
import sys
import traceback
from pathlib import Path

if __package__:
    from .external_tts import EXTERNAL_ENGINES
else:
    from external_tts import EXTERNAL_ENGINES


def import_engine(engine, source):
    sys.path.insert(0, str(source))
    if engine == 'cosyvoice':
        sys.path.insert(0, str(source / 'third_party' / 'Matcha-TTS'))
        import torch
        import onnxruntime
        # Let ORT use the CUDA/cuDNN libraries supplied by PyTorch's wheels.
        if hasattr(onnxruntime, 'preload_dlls'):
            onnxruntime.preload_dlls()
        from cosyvoice.cli.cosyvoice import CosyVoice3
        return CosyVoice3
    if engine == 'qwen3':
        from qwen_tts import Qwen3TTSModel
        return Qwen3TTSModel
    from indextts.infer_v2_5 import IndexTTS2
    return IndexTTS2


def load_model(engine, cls, model_dir, device):
    import torch
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in the selected TTS environment.')
    if engine == 'cosyvoice':
        return cls(model_dir=str(model_dir), fp16=device.startswith('cuda'))
    if engine == 'qwen3':
        return cls.from_pretrained(str(model_dir), device_map=device,
                                  dtype=torch.bfloat16 if device.startswith('cuda') else torch.float32,
                                  attn_implementation='sdpa')
    return cls(cfg_path=str(model_dir / 'config.yaml'), model_dir=str(model_dir),
               device=device, use_bf16=device.startswith('cuda'), use_cuda_kernel=False,
               use_deepspeed=False, use_accel=False, use_qwen_emo=False)


def synthesize(engine, model, request):
    import numpy as np
    import soundfile as sf
    output = request['output']
    Path(output).unlink(missing_ok=True)
    if engine == 'qwen3':
        wavs, rate = model.generate_custom_voice(text=request['text'],
                                                language=request['language'],
                                                speaker=request['speaker'])
        sf.write(output, wavs[0], rate, subtype='FLOAT')
    elif engine == 'cosyvoice':
        import torch
        chunks = [item['tts_speech'].detach().cpu() for item in model.inference_cross_lingual(
            'You are a helpful assistant.<|endofprompt|>' + request['text'], request['voice'], stream=False)]
        if not chunks:
            raise RuntimeError('CosyVoice returned no audio.')
        rate = model.sample_rate
        sf.write(output, torch.cat(chunks, dim=-1).squeeze(0).numpy(), rate, subtype='FLOAT')
    else:
        model.infer(spk_audio_prompt=request['voice'], text=request['text'], output_path=output,
                    lang=request['language'], verbose=False)
        rate = sf.info(output).samplerate
    data, rate = sf.read(output, dtype='float32')
    if not data.size or not np.isfinite(data).all():
        raise RuntimeError('TTS returned empty or invalid audio.')
    return rate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', choices=EXTERNAL_ENGINES, required=True)
    parser.add_argument('--source_dir', type=Path, required=True)
    parser.add_argument('--model_dir', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--check-imports', action='store_true')
    args = parser.parse_args()
    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    # Redirect Python and native library output before any model imports.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), 'w', encoding='utf-8', buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr

    def reply(**message):
        protocol.write(json.dumps(message, ensure_ascii=False) + '\n')
        protocol.flush()

    try:
        cls = import_engine(args.engine, args.source_dir)
        if args.check_imports:
            if args.engine == 'cosyvoice' and args.model_dir:
                # HyperPyYAML imports these at model load time. Check the symbols
                # without constructing networks or loading their checkpoints.
                config = (args.model_dir / 'cosyvoice3.yaml').read_text()
                for symbol in set(re.findall(r'!(?:new|name|apply):([\w.]+)', config)):
                    module, attribute = symbol.rsplit('.', 1)
                    getattr(importlib.import_module(module), attribute)
            import torch
            if args.device == 'cuda':
                torch.ones((16, 16), device='cuda').matmul(torch.ones((16, 16), device='cuda'))
                torch.cuda.synchronize()
            reply(ok=True, engine=args.engine, torch=torch.__version__, cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name() if args.device == 'cuda' else None)
            return
        if args.model_dir is None:
            raise ValueError('--model_dir is required for synthesis.')
        model = load_model(args.engine, cls, args.model_dir, args.device)
        reply(ok=True, samplerate=EXTERNAL_ENGINES[args.engine]['samplerate'])
        for line in sys.stdin:
            try:
                rate = synthesize(args.engine, model, json.loads(line))
                reply(ok=True, samplerate=rate)
            except Exception as error:
                traceback.print_exc()
                reply(ok=False, error=f'{type(error).__name__}: {error}')
    except Exception as error:
        traceback.print_exc()
        reply(ok=False, error=f'{type(error).__name__}: {error}')
        raise SystemExit(1)


if __name__ == '__main__':
    main()
