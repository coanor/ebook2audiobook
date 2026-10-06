"""Shared paths and process protocol for TTS projects with separate dependencies."""
import atexit
import json
import os
import subprocess
import threading
from pathlib import Path


QWEN_SPEAKERS = ('Uncle_Fu', 'Vivian', 'Serena', 'Dylan', 'Eric', 'Ryan', 'Aiden', 'Ono_Anna', 'Sohee')
EXTERNAL_ENGINES = {
    'cosyvoice': {
        'label': 'COSYVOICE', 'source': 'CosyVoice', 'model': 'CosyVoice3',
        'samplerate': 24000, 'required': ('cosyvoice3.yaml', 'llm.pt', 'flow.pt', 'hift.pt', 'speech_tokenizer_v3.onnx'),
        'languages': {'zho': 'zh', 'eng': 'en', 'jpn': 'ja', 'kor': 'ko', 'deu': 'de', 'spa': 'es', 'fra': 'fr', 'ita': 'it', 'rus': 'ru'},
    },
    'qwen3': {
        'label': 'QWEN3', 'source': 'Qwen3-TTS', 'model': 'Qwen3-TTS-1.7B-CustomVoice',
        'samplerate': 24000, 'required': ('config.json', 'model.safetensors', 'speech_tokenizer/model.safetensors'),
        'languages': {'zho': 'Chinese', 'eng': 'English', 'jpn': 'Japanese', 'kor': 'Korean', 'deu': 'German', 'spa': 'Spanish', 'fra': 'French', 'ita': 'Italian', 'rus': 'Russian', 'por': 'Portuguese'},
    },
    'indextts': {
        'label': 'INDEXTTS', 'source': 'IndexTTS', 'model': 'IndexTTS-2.5',
        'samplerate': 22050, 'required': ('config.yaml', 'gpt.pth', 's2mel.pth', 'codec.pth', 'hf_cache/w2v-bert-2.0/model.safetensors', 'hf_cache/bigvgan/bigvgan_generator.pt', 'hf_cache/campplus_cn_common.bin'),
        'languages': {'zho': 'ZH', 'eng': 'EN', 'jpn': 'JA', 'kor': 'KO', 'deu': 'DE', 'spa': 'ES', 'fra': 'FR', 'ita': 'IT', 'rus': 'RU', 'por': 'PT'},
    },
}


def engine_paths(engine, runtime_dir, model_dir=None):
    config = EXTERNAL_ENGINES[engine]
    root = Path(runtime_dir).resolve()
    return {
        'source': root / 'components' / 'external-tts' / config['source'],
        'model': Path(model_dir).resolve() if model_dir else root / 'models' / 'tts' / 'external' / config['model'],
        'python': root / 'run' / 'external-tts' / engine / 'bin' / 'python',
        'reference': root / 'components' / 'external-tts' / 'CosyVoice' / 'asset' / 'zero_shot_prompt.wav',
    }


class EngineWorker:
    """Keep one model loaded, exchange JSON requests, and stop on close or exit."""

    def __init__(self, command, cwd=None, env=None):
        self.process = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, text=True, encoding='utf-8',
                                        bufsize=1, start_new_session=True)
        self.lock = threading.Lock()
        atexit.register(self.close)
        try:
            self.info = self._receive()
        except BaseException:
            self.close()
            raise

    def _receive(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('TTS worker exited unexpectedly; see its error above.')
        try:
            response = json.loads(line)
        except ValueError as error:
            raise RuntimeError(f'Invalid TTS worker response: {line[:200]}') from error
        if not response.get('ok'):
            raise RuntimeError(response.get('error', 'TTS worker failed'))
        return response

    def synthesize(self, text, output, voice, language, speaker):
        with self.lock:
            try:
                self.process.stdin.write(json.dumps({'text': text, 'output': os.fspath(output),
                                                     'voice': voice, 'language': language,
                                                     'speaker': speaker}, ensure_ascii=False) + '\n')
                self.process.stdin.flush()
                return self._receive()
            except BaseException:
                self.close()
                raise

    def close(self):
        atexit.unregister(self.close)
        process = getattr(self, 'process', None)
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for stream in (process.stdin, process.stdout):
            if stream:
                stream.close()
